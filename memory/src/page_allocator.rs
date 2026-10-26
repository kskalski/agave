use {
    crate::{
        AllocOptions, HUGE_PAGE_SIZE, HugePages, Remainder,
        mmap::{map, unmap},
    },
    allocator_api2::alloc::{AllocError, Allocator, Global, Layout},
    std::ptr::NonNull,
};

/// Allocator backing allocations with huge pages or regular pages.
///
/// It allocates directly from the kernel when [`AllocOptions`] selects huge page backing for an
/// allocation, which by default covers allocations spanning at least one huge page. Everything else
/// comes from [`Global`].
#[derive(Clone, Copy, Debug, Default)]
pub struct PageAllocator {
    options: AllocOptions,
}

impl PageAllocator {
    pub const fn new(options: AllocOptions) -> Self {
        Self { options }
    }

    fn transparent_remainder(&self, layout: Layout) -> Option<Remainder> {
        let HugePages::Transparent(remainder) = self.options.huge_pages else {
            return None;
        };
        (cfg!(target_os = "linux") && layout.size() >= HUGE_PAGE_SIZE).then_some(remainder)
    }

    fn allocate_transparent(
        layout: Layout,
        remainder: Remainder,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // `AllocError` carries no cause, so log the OS error before dropping it.
        let ptr = map(layout, remainder).map_err(|error| {
            log::warn!("mmap of {} bytes failed: {error}", layout.size());
            AllocError
        })?;
        Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
    }

    fn allocate_with(
        &self,
        layout: Layout,
        global: impl FnOnce(Layout) -> Result<NonNull<[u8]>, AllocError>,
    ) -> Result<NonNull<[u8]>, AllocError> {
        match self.transparent_remainder(layout) {
            Some(remainder) => Self::allocate_transparent(layout, remainder),
            None => global(layout),
        }
    }
}

unsafe impl Allocator for PageAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.allocate_with(layout, |layout| Global.allocate(layout))
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        // New mappings are zeroed, so skip the default memset, which would fault in every page.
        self.allocate_with(layout, |layout| Global.allocate_zeroed(layout))
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: `allocate` chose the same path for this layout.
        unsafe {
            match self.transparent_remainder(layout) {
                Some(_) => unmap(ptr, layout),
                None => Global.deallocate(ptr, layout),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use {super::*, allocator_api2::vec::Vec};

    #[test]
    fn test_round_trip_through_growth() {
        for alloc in [
            PageAllocator::new(AllocOptions::new().huge_pages(HugePages::Off)),
            PageAllocator::new(
                AllocOptions::new().huge_pages(HugePages::Transparent(Remainder::AtEnd)),
            ),
            PageAllocator::new(
                AllocOptions::new().huge_pages(HugePages::Transparent(Remainder::AtStart)),
            ),
        ] {
            let mut vec = Vec::with_capacity_in(HUGE_PAGE_SIZE / 16, alloc);
            vec.extend(0..(HUGE_PAGE_SIZE / 2) as u64);
            assert!(vec.iter().enumerate().all(|(i, &v)| v == i as u64));
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_huge_page_placement() {
        let layout = Layout::from_size_align(HUGE_PAGE_SIZE + 1, 1).unwrap();
        for (huge_pages, len) in [
            (HugePages::Transparent(Remainder::AtEnd), layout.size()),
            (HugePages::Transparent(Remainder::AtStart), layout.size()),
        ] {
            let alloc = PageAllocator::new(AllocOptions::new().huge_pages(huge_pages));
            let ptr = alloc.allocate_zeroed(layout).unwrap();
            assert_eq!(ptr.len(), len);
            let start = ptr.cast::<u8>().addr().get();
            let boundary = match huge_pages {
                HugePages::Transparent(Remainder::AtStart) => {
                    (start + len).next_multiple_of(crate::page_size())
                }
                _ => start,
            };
            assert!(boundary.is_multiple_of(HUGE_PAGE_SIZE));
            // SAFETY: `alloc` returned `ptr` for `layout`, and nothing else references it.
            unsafe {
                let bytes = ptr.as_ptr().as_mut().unwrap();
                assert!(bytes.iter().all(|&b| b == 0));
                bytes.fill(1);
                alloc.deallocate(ptr.cast(), layout);
            }
        }
    }
}
