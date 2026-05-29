use {
    crate::{FileInfo, file_io::FileOpener},
    agave_io_uring::{Completion, FixedSlab, Ring, RingAccess, RingOp},
    io_uring::{IoUring, opcode, squeue, types},
    libc::{O_NOATIME, O_NOFOLLOW, O_RDONLY},
    std::{
        ffi::CString,
        fs::File,
        io, mem,
        os::fd::{AsRawFd, FromRawFd as _, RawFd},
        path::{Path, PathBuf},
        pin::Pin,
        sync::Arc,
        time::Duration,
    },
};

// Sanity limit for slab size and number of concurrent operations.
const MAX_IN_PROGRESS_PATHS: usize = 1024;

// We shouldn't use too many threads, as they will contend a lot to lock the directory inode
const MAX_IOWQ_WORKERS: u32 = 2;

const CHECK_PROGRESS_AFTER_SUBMIT_TIMEOUT: Option<Duration> = Some(Duration::from_millis(10));

/// Multiple files opener with `io_uring` queue for open + statx operations.
pub struct IoUringFileOpener<'a> {
    ring: Ring<FileOpenerState<'a>, FileOpenerOp>,
}

impl<'a> IoUringFileOpener<'a> {
    /// Create a new `IoUringFileOpener` with provided `file_open` callback to notify the result.
    ///
    /// `file_open` callback receives `FileInfo` for requested paths with opened `File` and additional
    /// information obtained through `statx` kernel operation.
    pub fn new<F: FnMut(FileInfo) + 'a>(file_open: F) -> io::Result<Self> {
        // Each in-progress path will generate 2 operations, let submission queue hold half of them
        // before we explicitly syscall to submit them for execution (lets kernel start processing before
        // we fill in-progress slab, but also amortizes number of `submit` syscalls made).
        let io_uring = IoUring::builder().build(MAX_IN_PROGRESS_PATHS as u32)?;
        // Maximum number of spawned [bounded IO, unbounded IO] kernel threads, we don't expect
        // any unbounded work, but limit it to 1 just in case (0 leaves it unlimited).
        io_uring
            .submitter()
            .register_iowq_max_workers(&mut [MAX_IOWQ_WORKERS, 1])?;

        let state = FileOpenerState::new(file_open);
        let ring = Ring::new(io_uring, state);

        Ok(Self { ring })
    }
}

impl FileOpener for IoUringFileOpener<'_> {
    fn schedule_open_at_dir(
        &mut self,
        path: PathBuf,
        parent_dir_handle: Arc<File>,
    ) -> io::Result<()> {
        let file = PendingFile::from_path(path);
        // Safety: raw pointer to C-path is passed to operations, so `file` must remain in slab until
        // both of them are completed, this is ensured by calling `self.complete_file` only when
        // `file.try_take_completed_file_info` returns `Some` thus when both are done providing data.
        let path_cstring = file.path_cstring.as_ptr();

        let file_key = self.wait_add_file(file)?;

        let stat_op = FileOpenerOp::Statx(StatxOp::new(
            file_key,
            parent_dir_handle.clone(),
            path_cstring,
        ));
        self.ring.push(stat_op)?;

        let open_op = FileOpenerOp::Open(OpenOp {
            file_key,
            dir_handle: parent_dir_handle,
            path_cstring,
        });
        self.ring.push(open_op)
    }

    fn drain(&mut self) -> io::Result<()> {
        self.ring.drain()
    }
}

impl IoUringFileOpener<'_> {
    fn wait_add_file(&mut self, file: PendingFile) -> io::Result<usize> {
        loop {
            self.ring.process_completions()?;
            if self.ring.context().files.len() < self.ring.context().files.capacity() {
                break;
            }
            self.ring
                .submit_and_wait(1, CHECK_PROGRESS_AFTER_SUBMIT_TIMEOUT)?;
        }
        let file_key = self.ring.context_mut().files.insert(file);
        Ok(file_key)
    }
}

struct FileOpenerState<'a> {
    files: FixedSlab<PendingFile>,
    /// Externally provided callback to be called on opened files
    file_open: Box<dyn FnMut(FileInfo) + 'a>,
}

impl<'a> FileOpenerState<'a> {
    fn new(file_open: impl FnMut(FileInfo) + 'a) -> Self {
        Self {
            files: FixedSlab::with_capacity(MAX_IN_PROGRESS_PATHS),
            file_open: Box::new(file_open),
        }
    }

    fn mark_file_opened(&mut self, file_key: usize, fd: types::Fd) {
        let file = self.files.get_mut(file_key).unwrap();
        // Safety: we just received FD from io_uring open, so it's valid, track it in owned File
        file.open_file = Some(unsafe { File::from_raw_fd(fd.0) });
        if let Some(file_info) = file.try_take_completed_file_info() {
            self.complete_file(file_key, file_info)
        }
    }

    fn store_file_stat(&mut self, file_key: usize, statxbuf: libc::statx) {
        let file = self.files.get_mut(file_key).unwrap();
        file.size_from_statx = Some(statxbuf.stx_size);
        if let Some(file_info) = file.try_take_completed_file_info() {
            self.complete_file(file_key, file_info)
        }
    }

    fn complete_file(&mut self, file_key: usize, file_info: FileInfo) {
        self.file_open.as_mut()(file_info);
        self.files.remove(file_key);
    }
}

#[derive(Debug)]
struct OpenOp {
    dir_handle: Arc<File>,
    path_cstring: *const libc::c_char,
    file_key: usize,
}

impl OpenOp {
    fn entry(&mut self) -> squeue::Entry {
        let at_dir_fd = types::Fd(self.dir_handle.as_raw_fd());
        opcode::OpenAt::new(at_dir_fd, self.path_cstring)
            .flags(O_NOFOLLOW | O_NOATIME | O_RDONLY)
            .build()
    }

    fn complete(
        &mut self,
        ring: &mut Completion<FileOpenerState, FileOpenerOp>,
        res: io::Result<RawFd>,
    ) -> io::Result<()>
    where
        Self: Sized,
    {
        let fd = types::Fd(res?);
        ring.context_mut().mark_file_opened(self.file_key, fd);
        Ok(())
    }
}

#[derive(Debug)]
struct StatxOp {
    file_key: usize,
    dir_handle: Arc<File>,
    path_cstring: *const libc::c_char,
    statxbuf: libc::statx,
}

impl StatxOp {
    fn new(file_key: usize, dir_handle: Arc<File>, path_cstring: *const libc::c_char) -> Self {
        Self {
            file_key,
            dir_handle,
            path_cstring,
            statxbuf: unsafe { std::mem::zeroed() },
        }
    }

    fn entry(&mut self) -> squeue::Entry {
        let at_dir_fd = types::Fd(self.dir_handle.as_raw_fd());
        opcode::Statx::new(
            at_dir_fd,
            self.path_cstring,
            &mut self.statxbuf as *mut libc::statx as *mut types::statx,
        )
        .build()
    }

    fn complete(
        &mut self,
        ring: &mut Completion<FileOpenerState, FileOpenerOp>,
        res: io::Result<i32>,
    ) -> io::Result<()>
    where
        Self: Sized,
    {
        let _ = res?;
        ring.context_mut()
            .store_file_stat(self.file_key, self.statxbuf);
        Ok(())
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum FileOpenerOp {
    Open(OpenOp),
    Statx(StatxOp),
}

impl RingOp<FileOpenerState<'_>> for FileOpenerOp {
    fn entry(&mut self) -> squeue::Entry {
        match self {
            Self::Open(op) => op.entry(),
            Self::Statx(op) => op.entry(),
        }
    }

    fn complete(
        &mut self,
        ring: &mut Completion<FileOpenerState, Self>,
        res: io::Result<i32>,
    ) -> io::Result<()>
    where
        Self: Sized,
    {
        match self {
            Self::Open(op) => op.complete(ring, res),
            Self::Statx(op) => op.complete(ring, res),
        }
    }
}

/// In-progress state for file while open and stat operations progress
///
/// Once both file descriptor and statx information are available, then
/// `try_take_completed_file_info` can be used to consume the pending state into `FileInfo`
#[derive(Debug)]
struct PendingFile {
    path: PathBuf,
    path_cstring: Pin<CString>,
    open_file: Option<File>,
    size_from_statx: Option<u64>,
}

impl PendingFile {
    fn from_path(path: PathBuf) -> Self {
        let path_cstring = Self::path_cstring(&path);
        Self {
            path,
            path_cstring: Pin::new(path_cstring),
            open_file: None,
            size_from_statx: None,
        }
    }

    fn path_cstring(path: &Path) -> CString {
        let os_str = path.file_name().expect("path must contain filename");
        CString::new(os_str.as_encoded_bytes()).expect("path mustn't contain interior NULs")
    }

    fn try_take_completed_file_info(&mut self) -> Option<FileInfo> {
        let size = self.size_from_statx?;
        let file = self.open_file.take()?;
        let path = mem::take(&mut self.path);
        Some(FileInfo { file, size, path })
    }
}
