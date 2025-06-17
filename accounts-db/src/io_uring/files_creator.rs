use std::{
    collections::VecDeque,
    ffi::CStr,
    io::{self, Read},
    mem,
    os::{fd::RawFd, unix::ffi::OsStrExt as _},
    path::PathBuf,
    ptr,
    time::Duration,
};

use io_uring::{opcode, squeue, types, IoUring};
use libc::{O_CREAT, O_NOATIME, O_NOFOLLOW, O_TRUNC, O_WRONLY};
use slab::Slab;

use agave_io_uring::{Completion, Ring, RingOp};
use smallvec::SmallVec;

use crate::io_uring::memory::{IoFixedBuffer, LargeBuffer};

const DEFAULT_WRITE_SIZE: usize = 1024 * 1024;
#[allow(dead_code)]
const DEFAULT_BUFFER_SIZE: usize = 64 * DEFAULT_WRITE_SIZE;

const MAX_OPEN_FILES: usize = 1000;
const SQPOLL_IDLE_TIMEOUT: u32 = 50;
const MAX_IOWQ_WORKERS: u32 = 4;
const IO_URING_QUEUES_SIZE: u32 = 512;
const CHECK_PROGRESS_AFTER_SUBMIT_TIMEOUT: Option<Duration> = Some(Duration::from_millis(10));

/// Multiple files creator with `io_uring` queue for open -> write -> close
/// operations.
pub struct FilesCreator<F: FnMut(PathBuf), B = LargeBuffer> {
    ring: Ring<FileCreatorState<F>, FileCreatorOp<F>>,
    #[allow(dead_code)]
    backing_buffer: B,
}

struct FileCreatorState<F: FnMut(PathBuf)> {
    files: Slab<PendingFile>,
    buffers: VecDeque<IoFixedBuffer>,
    open_fds: usize,
    wrote_callback: F,
}

impl<F: FnMut(PathBuf)> FilesCreator<F, LargeBuffer> {
    /// Create a new `FilesCreator` using `wrote_callback` to notify caller when
    /// file contents are already persisted.
    ///
    /// See [FilesCreator::with_buffer] for more information.
    #[allow(dead_code)]
    pub fn new(wrote_callback: F) -> io::Result<Self> {
        Self::with_capacity(DEFAULT_BUFFER_SIZE, wrote_callback)
    }

    /// Create a new `FilesCreator` using internally allocated buffer of specified
    /// `buf_size` and default write size.
    pub fn with_capacity(buf_size: usize, wrote_callback: F) -> io::Result<Self> {
        Self::with_buffer(
            LargeBuffer::new(buf_size),
            DEFAULT_WRITE_SIZE,
            wrote_callback,
        )
    }
}

impl<B: AsMut<[u8]>, F: FnMut(PathBuf)> FilesCreator<F, B> {
    /// Create a new `FilesCreator` using provided `buffer` and `wrote_callback`
    /// to notify caller when file contents are already persisted.
    ///
    /// `buffer` is the internal buffer used for writing scheduled file contents.
    /// It must be at least `write_capacity` long. The creator will execute multiple
    /// `write_capacity` sized writes in parallel to empty the work queue of files to create.
    pub fn with_buffer(buffer: B, write_capacity: usize, wrote_callback: F) -> io::Result<Self> {
        let ring = IoUring::builder()
            .setup_coop_taskrun()
            .setup_sqpoll(SQPOLL_IDLE_TIMEOUT)
            .build(IO_URING_QUEUES_SIZE)?;
        ring.submitter()
            .register_iowq_max_workers(&mut [MAX_IOWQ_WORKERS, 0])?;
        Self::with_buffer_and_ring(ring, buffer, write_capacity, wrote_callback)
    }

    fn with_buffer_and_ring(
        ring: IoUring,
        mut backing_buffer: B,
        write_capacity: usize,
        wrote_callback: F,
    ) -> io::Result<Self> {
        let buffer = backing_buffer.as_mut();
        assert!(buffer.len() % write_capacity == 0);

        let buffers =
            IoFixedBuffer::register_and_chunk_buffer(&ring, buffer, write_capacity)?.collect();

        let ring = Ring::new(
            ring,
            FileCreatorState {
                files: Slab::with_capacity(MAX_OPEN_FILES),
                open_fds: 0,
                buffers,
                wrote_callback,
            },
        );

        Ok(Self {
            ring,
            backing_buffer,
        })
    }

    fn wait_free_buf(&mut self) -> io::Result<IoFixedBuffer> {
        loop {
            self.ring.process_completions()?;
            let buf = self.ring.context_mut().buffers.pop_front();
            if let Some(buf) = buf {
                return Ok(buf);
            }
            self.ring
                .submit_and_wait(1, CHECK_PROGRESS_AFTER_SUBMIT_TIMEOUT)?;
        }
    }

    pub fn drain(mut self) -> io::Result<()> {
        self.ring.drain()
    }

    pub fn wrote_callback(&mut self) -> &mut F {
        &mut self.ring.context_mut().wrote_callback
    }
}

impl<F: FnMut(PathBuf)> FilesCreator<F> {
    /// Schedule creating a file at `path` with `mode` permissons and
    /// bytes read from `contents`.
    pub fn create(&mut self, path: PathBuf, mode: u32, contents: impl Read) -> io::Result<()> {
        let file_key = self.open(path, mode)?;
        self.write_and_close(contents, file_key)
    }

    /// Schedule opening file at `path` with `mode` permissons.
    ///
    /// Returns key that can be used for scheduling writes for it.
    fn open(&mut self, path: PathBuf, mode: u32) -> io::Result<usize> {
        while self.ring.context().files.len() >= MAX_OPEN_FILES {
            eprintln!("too many open files");
            self.ring.process_completions()?;
            self.ring
                .submit_and_wait(1, CHECK_PROGRESS_AFTER_SUBMIT_TIMEOUT)?;
        }

        // FIXME: pre-open accounts/, change accounts/bla to bla so we don't
        // keep re-walking the path and locking etc

        let mut path_bytes = Vec::with_capacity(4096);
        let buf_ptr = path_bytes.as_mut_ptr() as *mut u8;
        let bytes = path.as_os_str().as_bytes();
        assert!(bytes.len() <= path_bytes.capacity() - 1);
        // Safety:
        // We know that the buffer is large enough to hold the copy and the
        // pointers don't overlap.
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), buf_ptr, bytes.len());
            buf_ptr.add(bytes.len()).write(0);
            path_bytes.set_len(bytes.len() + 1);
        }

        let file = PendingFile {
            path,
            fd: None,
            backlog: SmallVec::new(),
            writes_started: 0,
            writes_completed: 0,
            eof: false,
        };
        let file_key = self.ring.context_mut().files.insert(file);

        let op = FileCreatorOp::Open(OpenOp {
            path: path_bytes,
            mode,
            file_key,
            _f: std::marker::PhantomData,
        });
        self.ring.push(op)?;

        Ok(file_key)
    }

    fn write_and_close(&mut self, mut src: impl Read, file_key: usize) -> io::Result<()> {
        let mut offset = 0;
        loop {
            let mut buf = self.wait_free_buf()?;
            let state = self.ring.context_mut();
            let file = &mut state.files[file_key];

            let len = src.read(buf.as_mut())?;
            if len == 0 {
                file.eof = true;

                if len == 0 {
                    state.buffers.push_front(buf);
                    if file.complete() {
                        let path = mem::replace(&mut file.path, PathBuf::new());
                        (state.wrote_callback)(path);
                        let fd = file.fd.take().unwrap();
                        self.ring
                            .push(FileCreatorOp::Close(CloseOp::new(fd, file_key)))?;
                    }
                    break;
                }
            }

            file.writes_started += 1;
            if let Some(fd) = &file.fd {
                let op = WriteOp {
                    file_key,
                    fd: *fd,
                    offset,
                    buf,
                    write_len: len,
                    _f: std::marker::PhantomData,
                };

                self.ring.push(FileCreatorOp::Write(op))?;
            } else {
                file.backlog.push((buf, offset, len));
            }

            offset += len;
        }

        Ok(())
    }
}

struct OpenOp<F> {
    // FIXME: pin this
    path: Vec<u8>,
    mode: libc::mode_t,
    file_key: usize,
    _f: std::marker::PhantomData<F>,
}

impl<F> std::fmt::Debug for OpenOp<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenOp")
            .field("path", &unsafe { CStr::from_ptr(self.path.as_ptr() as _) })
            .field("file_key", &self.file_key)
            .finish()
    }
}

impl<F: FnMut(PathBuf)> OpenOp<F> {
    fn entry(&mut self) -> squeue::Entry {
        opcode::OpenAt::new(types::Fd(libc::AT_FDCWD), self.path.as_ptr() as _)
            .flags(O_CREAT | O_TRUNC | O_NOFOLLOW | O_WRONLY | O_NOATIME)
            .mode(self.mode)
            .build()
    }

    fn complete(
        &mut self,
        ring: &mut Completion<FileCreatorState<F>, FileCreatorOp<F>>,
        res: io::Result<i32>,
    ) -> io::Result<()>
    where
        Self: Sized,
    {
        let fd = match res {
            Ok(fd) => fd,
            Err(err) if err.kind() == io::ErrorKind::NotFound => {
                // Safety:
                // Self::path is guaranteed to be valid while it's referenced by pointer by the
                // corresponding squeue::Entry.
                eprintln!(
                    "retrying {}",
                    CStr::from_bytes_until_nul(&self.path)
                        .unwrap()
                        .to_string_lossy()
                );
                ring.push(FileCreatorOp::Open(Self {
                    path: std::mem::take(&mut self.path),
                    ..*self
                }));
                return Ok(());
            }
            Err(e) => return Err(e),
        };

        let state = ring.context_mut();
        let file = &mut state.files[self.file_key];
        // Safety: the fd is valid, having just been returned by the ring
        file.fd = Some(fd);
        state.open_fds += 1;

        let fd = file.fd.clone();
        let mut backlog = mem::replace(&mut file.backlog, SmallVec::new());
        for (buf, offset, len) in backlog.drain(..) {
            let op = WriteOp {
                file_key: self.file_key,
                fd: fd.clone().unwrap(),
                offset,
                buf,
                write_len: len,
                _f: std::marker::PhantomData,
            };
            ring.push(FileCreatorOp::Write(op));
        }

        Ok(())
    }
}

#[derive(Debug)]
struct CloseOp<F> {
    fd: Option<RawFd>,
    file_key: usize,
    _f: std::marker::PhantomData<F>,
}

impl<F: FnMut(PathBuf)> CloseOp<F> {
    fn new(fd: RawFd, file_key: usize) -> Self {
        Self {
            fd: Some(fd),
            file_key,
            _f: std::marker::PhantomData,
        }
    }

    fn entry(&mut self) -> squeue::Entry {
        opcode::Close::new(types::Fd(self.fd.take().unwrap())).build()
    }

    fn complete(
        &mut self,
        ring: &mut Completion<FileCreatorState<F>, FileCreatorOp<F>>,
        res: io::Result<i32>,
    ) -> io::Result<()>
    where
        Self: Sized,
    {
        let _ = res?;

        let state = ring.context_mut();
        let _ = state.files.remove(self.file_key);
        state.open_fds -= 1;
        Ok(())
    }
}

#[derive(Debug)]
struct WriteOp<F> {
    file_key: usize,
    fd: RawFd,
    offset: usize,
    buf: IoFixedBuffer,
    write_len: usize,
    _f: std::marker::PhantomData<F>,
}

impl<F: FnMut(PathBuf)> WriteOp<F> {
    fn entry(&mut self) -> squeue::Entry {
        let WriteOp {
            file_key: _,
            fd,
            offset,
            buf,
            write_len,
            _f: _,
        } = self;

        opcode::WriteFixed::new(
            types::Fd(*fd),
            buf.as_mut_ptr(),
            *write_len as u32,
            buf.io_buf_index(),
        )
        .offset(*offset as u64)
        .ioprio(2 << 13)
        .build()
        .flags(squeue::Flags::ASYNC)
    }

    fn complete(
        &mut self,
        ring: &mut Completion<FileCreatorState<F>, FileCreatorOp<F>>,
        res: io::Result<i32>,
    ) -> io::Result<()>
    where
        Self: Sized,
    {
        let written = res? as usize;

        let WriteOp {
            file_key,
            fd,
            offset: _,
            ref mut buf,
            write_len,
            _f: _,
        } = self;
        let buf = std::mem::replace(buf, IoFixedBuffer::empty());

        assert_eq!(written, *write_len, "short write");

        let state = ring.context_mut();
        state.buffers.push_front(buf);

        let file = &mut state.files[*file_key];
        file.writes_completed += 1;
        if file.complete() {
            let path = mem::replace(&mut file.path, PathBuf::new());
            (state.wrote_callback)(path);
            ring.push(FileCreatorOp::Close(CloseOp::new(*fd, *file_key)));
        }

        Ok(())
    }
}

#[derive(Debug)]
enum FileCreatorOp<F> {
    Open(OpenOp<F>),
    Close(CloseOp<F>),
    Write(WriteOp<F>),
}

impl<F: FnMut(PathBuf)> RingOp<FileCreatorState<F>> for FileCreatorOp<F> {
    fn entry(&mut self) -> squeue::Entry {
        match self {
            Self::Open(op) => op.entry(),
            Self::Close(op) => op.entry(),
            Self::Write(op) => op.entry(),
        }
    }

    fn complete(
        &mut self,
        ring: &mut Completion<FileCreatorState<F>, Self>,
        res: io::Result<i32>,
    ) -> io::Result<()>
    where
        Self: Sized,
    {
        match self {
            Self::Open(op) => op.complete(ring, res),
            Self::Close(op) => op.complete(ring, res),
            Self::Write(op) => op.complete(ring, res),
        }
    }
}

struct PendingFile {
    path: PathBuf,
    fd: Option<RawFd>,
    backlog: SmallVec<[(IoFixedBuffer, usize, usize); 8]>,
    eof: bool,
    writes_started: usize,
    writes_completed: usize,
}

impl PendingFile {
    fn complete(&self) -> bool {
        self.eof && self.writes_started == self.writes_completed
    }
}

impl std::fmt::Debug for PendingFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingFile")
            .field("fd", &self.fd)
            .field("complete", &self.eof)
            .field("writes_started", &self.writes_started)
            .field("writes_completed", &self.writes_completed)
            .field("backlog", &self.backlog)
            .finish()
    }
}
