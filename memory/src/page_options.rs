/// Size and alignment of a transparent huge page with 4 KiB base pages, as on x86_64.
pub const HUGE_PAGE_SIZE: usize = 2 * 1024 * 1024;

pub fn page_size() -> usize {
    #[cfg(unix)]
    // SAFETY: sysconf has no preconditions.
    return unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
    #[cfg(not(unix))]
    4096
}

/// Selects whether and how an allocation gets huge pages.
///
/// Huge pages cut page faults and TLB misses in large buffers and tables. The kernel grants
/// transparent huge pages when THP is enabled in `always` or `madvise` mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HugePages {
    /// Allocates regular pages through [`Global`](allocator_api2::alloc::Global).
    Off,
    /// Backs the allocation with transparent huge pages, placing the part short of a whole huge
    /// page based on [`Remainder`].
    Transparent(Remainder),
}

/// Placement of the part of an allocation short of a whole huge page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Remainder {
    /// Puts the remainder on regular pages at the end (doesn't round up the allocation).
    ///
    /// Suits allocations accessed evenly or hottest at their start.
    AtEnd,
    /// Puts the remainder on regular pages at the start (doesn't round up the allocation).
    ///
    /// Suits allocations whose hot data sits in a small tail after a bulk sized in whole huge
    /// pages, since the regular pages then fall in the less used bulk.
    AtStart,
}

/// Backing that [`PageAllocator`](crate::PageAllocator) and [`AlignedBuffer`](crate::AlignedBuffer)
/// select for their allocations.
///
/// The default backs allocations that span at least one huge page with transparent huge pages,
/// with the remainder at the end.
#[derive(Clone, Copy, Debug)]
pub struct AllocOptions {
    pub(crate) huge_pages: HugePages,
}

impl AllocOptions {
    pub const fn new() -> Self {
        Self {
            huge_pages: HugePages::Transparent(Remainder::AtEnd),
        }
    }

    pub const fn huge_pages(mut self, huge_pages: HugePages) -> Self {
        self.huge_pages = huge_pages;
        self
    }
}

impl Default for AllocOptions {
    fn default() -> Self {
        Self::new()
    }
}
