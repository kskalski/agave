use {
    crate::io_uring::{
        memory::{IoFixedBuffer, LargeBuffer},
        IO_PRIO_BE_HIGHEST,
    },
    agave_io_uring::{Completion, Ring, RingOp},
    io_uring::{opcode, squeue, types, IoUring},
    std::{
        collections::VecDeque,
        fs::File,
        io::{self, BufRead, Read},
        marker::PhantomData,
        mem,
        os::fd::{AsRawFd, RawFd},
        path::Path,
    },
};

const DEFAULT_READ_SIZE: usize = 1024 * 1024;
#[allow(dead_code)]
const DEFAULT_BUFFER_SIZE: usize = 64 * DEFAULT_READ_SIZE;
const SQPOLL_IDLE_TIMEOUT: u32 = 50;
const MAX_IOWQ_WORKERS: u32 = 4;

/// Reader for non-seekable files.
///
/// Implements read-ahead using io_uring.
pub struct SequentialFileReader<'a, B> {
    // Note: state is tied to `backing_buffer` and contains unsafe pointer references to it
    inner: Ring<SequentialFileReaderState, ReadOp>,
    owned_files: VecDeque<File>,
    /// Owned buffer used across lifespan of `inner` (should get dropped last)
    #[allow(dead_code)]
    backing_buffer: B,
    _phantom: PhantomData<&'a ()>,
}

impl SequentialFileReader<'_, LargeBuffer> {
    #[allow(dead_code)]
    pub fn from_path(path: impl AsRef<Path>) -> io::Result<Self> {
        let mut this = Self::new()?;
        let file = std::os::unix::fs::OpenOptionsExt::custom_flags(
            std::fs::OpenOptions::new().read(true),
            libc::O_NOATIME,
        )
        .open(path)?;
        this.add_file(file)?;
        Ok(this)
    }

    /// Create a new `SequentialFileReader` for the given `path` using internally allocated
    /// large buffer and default read size.
    ///
    /// See [SequentialFileReader::with_buffer] for more information.
    #[allow(dead_code)]
    pub fn new() -> io::Result<Self> {
        Self::with_capacity(DEFAULT_BUFFER_SIZE)
    }

    /// Create a new `SequentialFileReader` for the given `path` using internally allocated
    /// buffer of specified `buf_size` and default read size.
    pub fn with_capacity(buf_size: usize) -> io::Result<Self> {
        Self::with_buffer(LargeBuffer::new(buf_size), DEFAULT_READ_SIZE)
    }
}

impl<B: AsMut<[u8]>> SequentialFileReader<'_, B> {
    /// Create a new `SequentialFileReader` for the given file using provided backing `buffer`.
    ///
    /// `buffer` is the internal buffer used for reading. It must be at least `read_capacity` long.
    /// The reader will execute multiple `read_capacity` sized reads in parallel to fill the buffer.
    pub fn with_buffer(mut buffer: B, read_capacity: usize) -> io::Result<Self> {
        let buf_len = buffer.as_mut().len();

        // Let submission queue hold half of buffers before we explicitly syscall
        // to submit them for reading.
        let ring_qsize = (buf_len / read_capacity / 2).max(1) as u32;
        let ring = IoUring::builder()
            .setup_sqpoll(SQPOLL_IDLE_TIMEOUT)
            .build(ring_qsize)?;
        ring.submitter()
            .register_iowq_max_workers(&mut [MAX_IOWQ_WORKERS, 0])?;
        Self::with_buffer_and_ring(buffer, ring, read_capacity)
    }

    /// Create a new `SequentialFileReader` for the given file, using a custom
    /// ring instance.
    fn with_buffer_and_ring(
        mut backing_buffer: B,
        ring: IoUring,
        read_capacity: usize,
    ) -> io::Result<Self> {
        let buffer = backing_buffer.as_mut();
        assert!(buffer.len() >= read_capacity, "buffer too small");
        assert!(
            buffer.len() % read_capacity == 0,
            "buffer size must be a multiple of read_capacity"
        );

        let buffers = IoFixedBuffer::register_and_chunk_buffer(&ring, buffer, read_capacity)?
            .map(ReadBufState::Uninit)
            .collect();

        let state = SequentialFileReaderState {
            buffers,
            files: VecDeque::new(),
            next_read_file_index: None,
            next_read_buf_index: 0,
        };
        Ok(Self {
            inner: Ring::new(ring, state),
            owned_files: VecDeque::new(),
            backing_buffer,
            _phantom: PhantomData,
        })
    }
}

impl<'a, B> SequentialFileReader<'a, B> {
    /// Add file reference to read. Starts reading the file as soon as a buffer is available.
    pub fn add_file(&mut self, file: File) -> io::Result<()> {
        self.add_file_by_fd(file.as_raw_fd(), None)?;
        self.owned_files.push_back(file);
        Ok(())
    }

    /// Add file reference to read. Starts reading the file as soon as a buffer is available.
    /// Lifetime of reference is tied to the reader's lifetime.
    #[allow(unused)]
    pub fn add_file_ref(&mut self, file: &'a File, limit_len: usize) -> io::Result<()> {
        self.add_file_by_fd(file.as_raw_fd(), Some(limit_len))
    }

    /// Caller must ensure that the file is not closed while the reader is using it.
    fn add_file_by_fd(&mut self, fd: RawFd, limit_len: Option<usize>) -> io::Result<()> {
        let state = self.inner.context_mut();

        state.files.push_back(FileState::new(fd, limit_len));

        if state.all_buffers_used() {
            // Just added file to backlog, no reads can be started yet.
            return Ok(());
        }

        // There are free buffers, so we can start reading the new file.
        state.next_read_file_index = Some(state.next_read_file_index.map_or(0, |idx| idx + 1));

        // Start reading as many buffers as necessary for queued files.
        while let Some(op) = self.inner.context_mut().next_read_op() {
            self.inner.push(op)?;
        }
        // Make sure work is started in case submission queue is large and we
        // never submitted work when adding buffers.
        self.inner.submit()
    }

    #[allow(unused)]
    pub fn move_to_next_file(&mut self) -> io::Result<()> {
        let state = self.inner.context_mut();

        let Some(file_state) = state.files.pop_front() else {
            return Ok(());
        };
        if let Some(next_file_index) = state.next_read_file_index.as_mut() {
            state.next_read_file_index = next_file_index.checked_sub(1);
        }

        if let Some(buf_index) = file_state.current_buf_index {
            let ReadBufState::Full {
                eof_pos: Some(_), ..
            } = &state.buffers[buf_index]
            else {
                panic!("cannot move from incompletely read buffer");
            };
        }

        if file_state.read_limit.is_none() {
            // This was unlimited file read, so no other files are queued,
            // but some ops might still be in-flight, drain them.
            self.inner.drain()?;
            self.inner.context_mut().reset_buffers();
            if !self.inner.context().files.is_empty() {
                self.inner.context_mut().next_read_file_index = Some(0);
            }

            // Start reading as many buffers as necessary for queued files.
            while let Some(op) = self.inner.context_mut().next_read_op() {
                self.inner.push(op)?;
            }
            // Make sure work is started in case submission queue is large and we
            // never submitted work when adding buffers.
            self.inner.submit()?
        }

        if self
            .owned_files
            .front()
            .is_some_and(|f| f.as_raw_fd() == file_state.raw_fd)
        {
            self.owned_files.pop_front();
        }
        Ok(())
    }
}

/// Holds the state of a single file being read.
struct FileState {
    raw_fd: RawFd,
    /// Limit file offset to read up to.
    read_limit: Option<usize>,
    /// Offset of the next byte to read from file
    offset: usize,
    /// Current buffer to consume data from (if file is already being read)
    current_buf_index: Option<usize>,
}

impl FileState {
    fn new(raw_fd: RawFd, read_limit: Option<usize>) -> Self {
        Self {
            raw_fd,
            read_limit,
            offset: 0,
            current_buf_index: None,
        }
    }

    /// Create new read operation into the `bufs` buffer at `index`.
    ///
    /// This is called at start and as soon as a buffer is fully consumed by BufRead::fill_buf().
    ///
    /// Reads [state.offset, state.offset + state.read_capacity) from the file into
    /// `bufs[index]`. Once a read is complete, ReadOp::complete(state) is called to update
    /// the state.
    fn next_read_op(&mut self, index: usize, bufs: &mut [ReadBufState]) -> Option<ReadOp> {
        let Self {
            current_buf_index,
            raw_fd,
            offset,
            read_limit,
        } = self;
        let left_to_read = if let Some(limit_offset) = read_limit {
            if *limit_offset == *offset {
                return None;
            }
            *limit_offset - *offset
        } else {
            usize::MAX
        };

        let ReadBufState::Uninit(buf) = mem::replace(&mut bufs[index], ReadBufState::Reading)
        else {
            unreachable!("buffer at {index} should be uninitialized")
        };

        let read_len = left_to_read.min(buf.len());
        let op = ReadOp {
            raw_fd: *raw_fd,
            buf,
            buf_offset: 0,
            file_offset: *offset,
            read_len,
            is_last_read: read_limit.is_some_and(|l| *offset + read_len >= l),
            reader_buf_index: index,
        };
        if current_buf_index.is_none() {
            *current_buf_index = Some(index);
        }
        // We always advance by `read_len`. If we get a short read, we submit a new
        // read for the remaining data. See ReadOp::complete().
        *offset += read_len;

        Some(op)
    }
}

/// Holds the state of the reader.
struct SequentialFileReaderState {
    buffers: Vec<ReadBufState>,
    files: VecDeque<FileState>,
    next_read_file_index: Option<usize>,
    next_read_buf_index: usize,
}

impl SequentialFileReaderState {
    /// The front buffer index (of `self.buffers`) where already read bytes of current file
    /// are available.
    fn current_buf_index(&self) -> Option<usize> {
        self.files.front().and_then(|f| f.current_buf_index)
    }

    fn current_buf(&self) -> Option<&ReadBufState> {
        self.current_buf_index().map(|idx| &self.buffers[idx])
    }

    fn current_buf_mut(&mut self) -> Option<&mut ReadBufState> {
        self.current_buf_index().map(|idx| &mut self.buffers[idx])
    }

    fn next_read_op(&mut self) -> Option<ReadOp> {
        if self.all_buffers_used() {
            return None;
        }
        loop {
            let read_file_index = self.next_read_file_index?;
            match self.files[read_file_index]
                .next_read_op(self.next_read_buf_index, &mut self.buffers)
            {
                Some(op) => {
                    self.next_read_buf_index = (self.next_read_buf_index + 1) % self.buffers.len();
                    return Some(op);
                }
                None => {
                    // Last read file reached its limit, try to move to the next file
                    if read_file_index < self.files.len() - 1 {
                        self.next_read_file_index = Some(read_file_index + 1);
                    } else {
                        return None;
                    }
                }
            }
        }
    }

    fn all_buffers_used(&mut self) -> bool {
        self.buffers[self.next_read_buf_index].is_used()
    }

    fn reset_buffers(&mut self) {
        for buf in &mut self.buffers {
            match buf {
                ReadBufState::Uninit(_) => (),
                ReadBufState::Reading => unreachable!("no buffer should be reading"),
                ReadBufState::Full {
                    buf: fixed_io_buf, ..
                } => {
                    *buf = ReadBufState::Uninit(mem::replace(fixed_io_buf, IoFixedBuffer::empty()));
                }
            }
        }
    }
}

// BufRead requires Read, but we never really use the Read interface.
impl<B: AsMut<[u8]>> Read for SequentialFileReader<'_, B> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let available = self.fill_buf()?;
        if available.is_empty() {
            return Ok(0); // EOF.
        }

        let bytes_to_read = available.len().min(buf.len());
        buf[..bytes_to_read].copy_from_slice(&available[..bytes_to_read]);
        self.consume(bytes_to_read);
        Ok(bytes_to_read)
    }
}

impl<B: AsMut<[u8]>> BufRead for SequentialFileReader<'_, B> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        let _have_data = loop {
            let state = self.inner.context_mut();
            let Some(file_state) = state.files.front_mut() else {
                return Ok(&[]);
            };
            let Some(current_buf_index) = file_state.current_buf_index.as_mut() else {
                return Ok(&[]);
            };
            match &mut state.buffers[*current_buf_index] {
                ReadBufState::Full { buf, pos, eof_pos } => {
                    if *pos < buf.len() && eof_pos.is_none_or(|eof| *pos < eof) {
                        // we have some data available
                        break true;
                    }

                    if eof_pos.is_some() {
                        // This is the last filled buf for the whole file
                        return Ok(&[]);
                    } else {
                        // We have finished consuming this buffer - uninitialize its state.
                        state.buffers[*current_buf_index] =
                            ReadBufState::Uninit(mem::replace(buf, IoFixedBuffer::empty()));
                        *current_buf_index = (*current_buf_index + 1) % state.buffers.len();

                        // Queue up next read
                        if let Some(op) = state.next_read_op() {
                            self.inner.push(op)?;
                        }
                    }

                    // move to the next buffer and check again whether we have data
                    continue;
                }
                ReadBufState::Uninit(_) => unreachable!("should be initialized"),
                _ => break false,
            }
        };

        loop {
            self.inner.process_completions()?;
            match self.inner.context().current_buf().unwrap() {
                ReadBufState::Full { .. } => break,
                ReadBufState::Uninit(_) => unreachable!("should be initialized"),
                // Still no data, wait for more completions.
                ReadBufState::Reading => _ = self.inner.submit_and_wait(1, None)?,
            }
        }

        // At this point we must have data or be at EOF.
        match self.inner.context().current_buf().unwrap() {
            ReadBufState::Full { buf, pos, eof_pos } => Ok(buf.as_slice(*pos, eof_pos)),
            // after the loop above we either have some data or we must be at EOF
            _ => unreachable!(),
        }
    }

    fn consume(&mut self, amt: usize) {
        let state = self.inner.context_mut();
        let Some(current_buf) = state.current_buf_mut() else {
            return;
        };
        match current_buf {
            ReadBufState::Full { pos, .. } => *pos += amt,
            _ => assert_eq!(amt, 0),
        }
    }
}

#[derive(Debug)]
enum ReadBufState {
    /// The buffer is pending submission to read queue (on initialization and
    /// in transition from `Full` to `Reading`).
    Uninit(IoFixedBuffer),
    /// The buffer is currently being read and there's a corresponding ReadOp in
    /// the ring.
    Reading,
    /// The buffer is filled and ready to be consumed.
    Full {
        buf: IoFixedBuffer,
        pos: usize,
        eof_pos: Option<usize>,
    },
}

impl ReadBufState {
    fn is_used(&self) -> bool {
        matches!(self, ReadBufState::Reading | ReadBufState::Full { .. })
    }
}

#[derive(Debug)]
struct ReadOp {
    raw_fd: RawFd,
    buf: IoFixedBuffer,
    /// This is the offset inside the buffer. It's typically 0, but can be non-zero if a previous
    /// read returned less data than requested (because of EINTR or whatever) and we submitted a new
    /// read for the remaining data.
    buf_offset: usize,
    /// The offset in the file.
    file_offset: usize,
    /// The length of the read. This is typically `read_capacity` but can be less if a previous read
    /// returned less data than requested.
    read_len: usize,
    /// Indicates that after reading `read_len` we have reached configured read limit.
    is_last_read: bool,
    /// This is the index of the buffer in the reader's state. It's used to update the state once the
    /// read completes.
    reader_buf_index: usize,
}

impl RingOp<SequentialFileReaderState> for ReadOp {
    fn entry(&mut self) -> squeue::Entry {
        let ReadOp {
            raw_fd: fd,
            buf,
            buf_offset: buf_off,
            file_offset: file_off,
            read_len,
            is_last_read: _,
            reader_buf_index: _,
        } = self;
        debug_assert!(*buf_off + *read_len <= buf.len());
        opcode::ReadFixed::new(
            types::Fd(*fd),
            // Safety: we assert that the buffer is large enough to hold the read.
            unsafe { buf.as_mut_ptr().byte_add(*buf_off) },
            *read_len as u32,
            buf.io_buf_index(),
        )
        .offset(*file_off as u64)
        .ioprio(IO_PRIO_BE_HIGHEST)
        .build()
    }

    fn complete(
        &mut self,
        completion: &mut Completion<SequentialFileReaderState, Self>,
        res: io::Result<i32>,
    ) -> io::Result<()> {
        let ReadOp {
            raw_fd: fd,
            buf,
            buf_offset: buf_off,
            file_offset: file_off,
            read_len,
            is_last_read,
            reader_buf_index,
        } = self;
        let reader_state = completion.context_mut();

        let last_read_len = res? as usize;

        let total_read_len = *buf_off + last_read_len;

        if last_read_len > 0 && last_read_len < *read_len {
            // Partial read, retry the op with updated offsets
            let op: ReadOp = ReadOp {
                raw_fd: *fd,
                buf: mem::replace(buf, IoFixedBuffer::empty()),
                buf_offset: total_read_len,
                file_offset: *file_off + last_read_len,
                read_len: *read_len - last_read_len,
                reader_buf_index: *reader_buf_index,
                is_last_read: *is_last_read,
            };
            // Safety:
            // The op points to a buffer which is guaranteed to be valid for the
            // lifetime of the operation
            completion.push(op);
        } else {
            reader_state.buffers[*reader_buf_index] = ReadBufState::Full {
                buf: mem::replace(buf, IoFixedBuffer::empty()),
                pos: 0,
                eof_pos: (last_read_len == 0 || *is_last_read).then_some(total_read_len),
            };
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use {super::*, tempfile::NamedTempFile};

    fn check_reading_file(file_size: usize, backing_buffer_size: usize, read_capacity: usize) {
        let pattern: Vec<u8> = (0..251).collect();

        // Create a temp file and write the pattern to it repeatedly
        let mut temp_file = NamedTempFile::new().unwrap();
        for _ in 0..file_size / pattern.len() {
            io::Write::write_all(&mut temp_file, &pattern).unwrap();
        }
        io::Write::write_all(&mut temp_file, &pattern[..file_size % pattern.len()]).unwrap();

        let buf = vec![0; backing_buffer_size];
        let mut reader = SequentialFileReader::with_buffer(buf, read_capacity).unwrap();
        reader
            .add_file(File::open(temp_file.path()).unwrap())
            .unwrap();

        // Read contents from the reader and verify length
        let mut all_read_data = Vec::new();
        reader.read_to_end(&mut all_read_data).unwrap();
        assert_eq!(all_read_data.len(), file_size);

        // Verify the contents
        for (i, byte) in all_read_data.iter().enumerate() {
            assert_eq!(*byte, pattern[i % pattern.len()], "Mismatch - pos {}", i);
        }
    }

    /// Test with buffer larger than the whole file
    #[test]
    fn test_reading_small_file() {
        check_reading_file(2500, 4096, 1024);
        check_reading_file(2500, 4096, 2048);
        check_reading_file(2500, 4096, 4096);
    }

    /// Test with buffer smaller than the whole file
    #[test]
    fn test_reading_file_in_chunks() {
        check_reading_file(25_000, 16384, 1024);
        check_reading_file(25_000, 4096, 1024);
        check_reading_file(25_000, 4096, 2048);
        check_reading_file(25_000, 4096, 4096);
    }

    /// Test with buffer much smaller than the whole file
    #[test]
    fn test_reading_large_file() {
        check_reading_file(250_000, 32768, 1024);
        check_reading_file(250_000, 16384, 1024);
        check_reading_file(250_000, 4096, 1024);
        check_reading_file(250_000, 4096, 2048);
        check_reading_file(250_000, 4096, 4096);
    }

    #[test]
    fn test_add_file_ref() {
        let mut temp_file = NamedTempFile::new().unwrap();
        io::Write::write_all(&mut temp_file, &[0xa, 0xb, 0xc]).unwrap();

        let buf = vec![0; 1024];
        let mut f = File::open(temp_file.path()).unwrap();
        {
            let mut reader = SequentialFileReader::with_buffer(buf, 512).unwrap();
            reader.add_file_ref(&f, 3).unwrap();
            let mut all_read_data = Vec::new();
            reader.read_to_end(&mut all_read_data).unwrap();
            assert_eq!(all_read_data.len(), 3);
        }
        let mut all_read_data = Vec::new();
        f.read_to_end(&mut all_read_data).unwrap();
        assert_eq!(all_read_data.len(), 3);
    }

    #[test]
    fn test_multiple_unlimited_files() {
        let mut temp_file1 = NamedTempFile::new().unwrap();
        io::Write::write_all(&mut temp_file1, &[0xa, 0xb, 0xc]).unwrap();

        let mut temp_file2 = NamedTempFile::new().unwrap();
        io::Write::write_all(&mut temp_file2, &[0xd, 0xe, 0xf, 0x10]).unwrap();

        let buf = vec![0; 1024];
        let f1 = File::open(temp_file1.path()).unwrap();
        let f2 = File::open(temp_file2.path()).unwrap();
        let mut reader = SequentialFileReader::with_buffer(buf, 512).unwrap();
        reader.add_file(f1).unwrap();
        reader.add_file(f2).unwrap();
        let mut all_read_data = Vec::new();
        reader.read_to_end(&mut all_read_data).unwrap();
        assert_eq!(all_read_data, vec![0xa, 0xb, 0xc]);

        reader.move_to_next_file().unwrap();
        all_read_data.clear();
        reader.read_to_end(&mut all_read_data).unwrap();
        assert_eq!(all_read_data, vec![0xd, 0xe, 0xf, 0x10]);
    }

    #[test]
    fn test_multiple_limited_files() {
        let mut temp_file1 = NamedTempFile::new().unwrap();
        io::Write::write_all(&mut temp_file1, &[0xa, 0xb, 0xc]).unwrap();

        let mut temp_file2 = NamedTempFile::new().unwrap();
        io::Write::write_all(&mut temp_file2, &[0xd, 0xe, 0xf, 0x10]).unwrap();

        let buf = vec![0; 1024];
        let f1 = File::open(temp_file1.path()).unwrap();
        let f2 = File::open(temp_file2.path()).unwrap();
        let mut reader = SequentialFileReader::with_buffer(buf, 512).unwrap();
        reader.add_file_ref(&f1, 2).unwrap();
        reader.add_file_ref(&f2, 3).unwrap();
        let mut all_read_data = Vec::new();
        reader.read_to_end(&mut all_read_data).unwrap();
        assert_eq!(all_read_data, vec![0xa, 0xb]);

        reader.move_to_next_file().unwrap();
        all_read_data.clear();
        reader.read_to_end(&mut all_read_data).unwrap();
        assert_eq!(all_read_data, vec![0xd, 0xe, 0xf]);
    }
}
