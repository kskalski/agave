use std::{
    fs::File,
    io::{self, BufRead, Cursor, Read},
    mem,
    os::fd::{AsRawFd as _, RawFd},
    slice,
};

use io_uring::{opcode, squeue, types, IoUring};

use crate::ring::{Ring, RingCtx, RingOp};

/// Reader for non-seekable files.
///
/// Implements read-ahead using io_uring.
pub struct SequentialFileReader<'a> {
    ring: Ring<SequentialFileReaderState<'a>, ReadOp<'a>>,
}

// Holds the state of the reader.
struct SequentialFileReaderState<'a> {
    file: File,
    read_capacity: usize,
    offset: usize,
    eof: bool,
    buffers: Vec<ReadBufState<'a>>,
    current_buf: usize,
}

impl<'a> SequentialFileReader<'a> {
    /// Create a new `SequentialFileReader` for the given file.
    ///
    /// `buffer` is the internal buffer used for reading. It must be at least `read_capacity` long.
    /// The reader will execute multiple `read_capacity` sized reads in parallel to fill the buffer.
    pub fn new(file: File, buffer: &'a mut [u8], read_capacity: usize) -> io::Result<Self> {
        let ring = IoUring::builder().setup_sqpoll(50).build(64).unwrap();
        ring.submitter()
            .register_iowq_max_workers(&mut [4, 0])
            .unwrap();
        Self::with_ring(ring, file, buffer, read_capacity)
    }

    /// Create a new `SequentialFileReader` for the given file, using a custom
    /// ring instance.
    ///
    /// See [SequentialFileReader::new] for more information.
    pub fn with_ring(
        ring: IoUring,
        file: File,
        buffer: &'a mut [u8],
        read_capacity: usize,
    ) -> io::Result<Self> {
        assert!(buffer.len() >= read_capacity, "buffer too small");
        assert!(
            buffer.len() % read_capacity == 0,
            "buffer size must be a multiple of read_capacity"
        );

        // We register fixed buffers in chunks of up to 1GB as this is faster than registering many
        // `read_capacity` buffers. Registering fixed buffers saves the kernel some work in
        // checking/mapping/unmapping buffers for each read operation.
        const FIXED_BUFFER_LEN: usize = 1024 * 1024 * 1024;
        let iovecs = buffer
            .chunks(FIXED_BUFFER_LEN)
            .map(|buf| libc::iovec {
                iov_base: buf.as_ptr() as _,
                iov_len: buf.len(),
            })
            .collect::<Vec<_>>();

        // Split the buffer into `read_capacity` sized chunks.
        let buf_start = buffer.as_ptr() as usize;
        let buffers = buffer
            .chunks_exact_mut(read_capacity)
            .map(|buf| {
                let io_buf_index = (buf.as_ptr() as usize - buf_start) / FIXED_BUFFER_LEN;
                ReadBufState::Empty { io_buf_index, buf }
            })
            // FIXME: remove this allocation, use const generics
            .collect::<Vec<_>>();
        println!("iovecs: {}, buffers: {}", iovecs.len(), buffers.len());

        let ring = Ring::new(
            ring,
            SequentialFileReaderState {
                file,
                read_capacity,
                buffers,
                offset: 0,
                eof: false,
                current_buf: 0,
            },
        );
        // Safety:
        // The iovecs point to a buffer which is guaranteed to be valid for the
        // lifetime of the reader
        unsafe { ring.register_buffers(&iovecs)? };

        let mut reader = Self { ring };

        // Start reading all buffers.
        for i in 0..reader.ring.ctx().buffers.len() {
            reader.start_reading_buf(i);
        }

        Ok(reader)
    }

    // Start reading into the buffer at `index`.
    //
    // This is called at start and as soon as a buffer is fully consumed by BufRead::fill_buf().
    //
    // Reads [state.offset, state.offset + state.read_capacity) from the file into
    // state.buffers[index]. Once a read is complete, ReadOp::complete(state) is called to update
    // the state.
    fn start_reading_buf(&mut self, index: usize) {
        let SequentialFileReaderState {
            buffers,
            current_buf: _,
            file,
            offset,
            read_capacity,
            eof: _,
        } = &mut self.ring.ctx_mut();
        let read_buf = mem::replace(&mut buffers[index], ReadBufState::Reading);
        match read_buf {
            ReadBufState::Empty { buf, io_buf_index } => {
                let op = ReadOp {
                    fd: file.as_raw_fd(),
                    buf,
                    buf_off: 0,
                    io_buf_index,
                    file_off: *offset,
                    read_len: *read_capacity,
                    reader_buf_index: index,
                };

                // We always advance by `read_capacity`. If we get a short read, we submit a new
                // read for the remaining data. See ReadOp::complete().
                *offset += *read_capacity;

                // Safety:
                // The op points to a buffer which is guaranteed to be valid for
                // the lifetime of the operation ('a)
                unsafe { self.ring.push(op).expect("pushing ReadOp failed") };
            }
            _ => unreachable!("called start_reading_buf on a non-empty buffer"),
        }
    }
}

// BufRead requires Read, but we never really use the Read interface.
impl<'a> Read for SequentialFileReader<'a> {
    fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> {
        unimplemented!("use BufRead, Read is slower");
    }
}

impl<'a> BufRead for SequentialFileReader<'a> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        println!("fill buff");
        let _have_data = loop {
            let state = self.ring.ctx_mut();
            let read_buf = &mut state.buffers[state.current_buf];
            match read_buf {
                ReadBufState::Full {
                    ref mut cursor,
                    io_buf_index,
                } => {
                    if !cursor.fill_buf()?.is_empty() {
                        // we have some data available
                        break true;
                    }

                    // we have finished consuming this buffer, queue the next read
                    let cursor = mem::replace(cursor, Cursor::new(&mut []));
                    let buf = cursor.into_inner();

                    // The very last read when we hit EOF could return less than `read_capacity`, in
                    // which case what's in the cursor is shorter than `read_capacity` and for
                    // strict correctness we reset the length.
                    //
                    // Note though that once we hit EOF we don't queue any more reads, so even if we
                    // didn't reset the length it wouldn't matter.
                    debug_assert!(buf.len() == state.read_capacity || state.eof);

                    state.buffers[state.current_buf] = ReadBufState::Empty {
                        // Safety: all buffers are created as being `read_capacty` large.
                        buf: unsafe {
                            slice::from_raw_parts_mut(buf.as_mut_ptr(), state.read_capacity)
                        },
                        io_buf_index: *io_buf_index,
                    };
                    let index = state.current_buf;
                    state.current_buf = (state.current_buf + 1) % state.buffers.len();
                    self.start_reading_buf(index);

                    // move to the next buffer and check again whether we have data
                    continue;
                }
                _ => break false,
            }
        };

        println!("check on io");
        loop {
            // FIXME: submit and wait
            self.ring.process_completions()?;
            let state = self.ring.ctx();

            match &state.buffers[state.current_buf] {
                ReadBufState::Full { .. } => break,
                ReadBufState::Empty { .. } if state.eof => break,
                // Still no data, wait for more completions.
                _ => {}
            }
        }
        println!("got data");

        // At this point we must have data or be at EOF.
        let state = self.ring.ctx_mut();
        match &mut state.buffers[state.current_buf] {
            ReadBufState::Full { cursor, .. } => Ok(cursor.fill_buf()?),
            ReadBufState::Empty { .. } if state.eof => Ok(&[]),
            // after the loop above we either have some data or we must be at EOF
            _ => unreachable!(),
        }
    }

    fn consume(&mut self, amt: usize) {
        let state = self.ring.ctx_mut();
        match &mut state.buffers[state.current_buf] {
            ReadBufState::Full { cursor, .. } => cursor.consume(amt),
            _ => assert_eq!(amt, 0),
        }
    }
}

enum ReadBufState<'a> {
    // The buffer is empty and ready to be filled.
    Empty {
        buf: &'a mut [u8],
        io_buf_index: usize,
    },
    // The buffer is currently being read and there's a corresponding ReadOp in
    // the ring.
    Reading,
    // The buffer is full and ready to be consumed.
    Full {
        cursor: Cursor<&'a mut [u8]>,
        io_buf_index: usize,
    },
}

impl<'a> std::fmt::Debug for ReadBufState<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty {
                buf: _,
                io_buf_index,
            } => f
                .debug_struct("Empty")
                .field("io_buf_index", io_buf_index)
                .finish(),
            Self::Reading => write!(f, "Reading"),
            Self::Full {
                cursor: _,
                io_buf_index,
            } => f
                .debug_struct("Full")
                .field("io_buf_index", io_buf_index)
                .finish(),
        }
    }
}

struct ReadOp<'a> {
    fd: RawFd,
    buf: &'a mut [u8],
    // This is the offset inside the buffer. It's typically 0, but can be non-zero if a previous
    // read returned less data than requested (because of EINTR or whatever) and we submitted a new
    // read for the remaining data.
    buf_off: usize,
    // The index of the fixed buffer in the ring. See register_buffers().
    io_buf_index: usize,
    // The offset in the file.
    file_off: usize,
    // The length of the read. This is typically `read_capacity` but can be less if a previous read
    // returned less data than requested.
    read_len: usize,
    // This is the index of the buffer in the reader's state. It's used to update the state once the
    // read completes.
    reader_buf_index: usize,
}

impl<'a> std::fmt::Debug for ReadOp<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReadOp")
            .field("fd", &self.fd)
            .field("buf_off", &self.buf_off)
            .field("io_buf_index", &self.io_buf_index)
            .field("file_off", &self.file_off)
            .field("read_len", &self.read_len)
            .field("reader_buf_index", &self.reader_buf_index)
            .finish()
    }
}

impl<'a> RingOp<SequentialFileReaderState<'a>> for ReadOp<'a> {
    fn entry(&mut self) -> squeue::Entry {
        let ReadOp {
            fd,
            buf,
            buf_off,
            io_buf_index,
            file_off,
            read_len,
            reader_buf_index: _,
        } = self;
        debug_assert!(*buf_off + *read_len <= buf.len());
        let prio =
            ioprio::Priority::new(ioprio::Class::BestEffort(ioprio::BePriorityLevel::highest()));
        opcode::ReadFixed::new(
            types::Fd(*fd),
            // Safety: we assert that the buffer is large enough to hold the read.
            unsafe { buf.as_mut_ptr().byte_add(*buf_off) },
            *read_len as u32,
            *io_buf_index as u16,
        )
        .offset(*file_off as u64)
        .ioprio(prio.inner())
        .build()
    }

    fn complete(
        self,
        res: io::Result<i32>,
        ring: &mut RingCtx<SequentialFileReaderState<'a>, Self>,
    ) -> io::Result<()> {
        let ReadOp {
            fd,
            buf,
            buf_off,
            io_buf_index,
            file_off,
            read_len,
            reader_buf_index,
        } = self;

        let reader_state = ring.ctx_mut();

        let last_read_len = res? as usize;
        if last_read_len == 0 {
            reader_state.eof = true;
        }

        let total_read_len = buf_off + last_read_len;

        if last_read_len < read_len && !reader_state.eof {
            println!(
                "SHORT READ {reader_buf_index} {} last {last_read_len} total {total_read_len} expected {read_len}",
                reader_state.current_buf
            );
            let op = ReadOp {
                fd,
                buf,
                buf_off: total_read_len,
                io_buf_index,
                file_off: file_off + last_read_len,
                read_len: read_len - last_read_len,
                reader_buf_index,
            };
            // Safety:
            // The op points to a buffer which is guaranteed to be valid for the
            // lifetime of the operation ('a)
            unsafe { ring.push(op).expect("pushing ReadOp failed") };
        } else {
            reader_state.buffers[reader_buf_index] = ReadBufState::Full {
                cursor: Cursor::new(&mut buf[..total_read_len]),
                io_buf_index,
            };
        }

        Ok(())
    }
}
