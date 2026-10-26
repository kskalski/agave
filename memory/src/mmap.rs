#[cfg(target_os = "linux")]
pub(crate) use linux::{map, map_reserved, unmap};

#[cfg(target_os = "linux")]
mod linux {
    use {
        crate::{HUGE_PAGE_SIZE, Remainder, page_size},
        std::{
            alloc::Layout,
            io,
            ptr::{self, NonNull},
            sync::Once,
        },
    };

    /// Maps zeroed memory for `layout` and advises transparent huge pages for it.
    ///
    /// `remainder` specifies the placement of the mapping part that doesn't span a whole huge page.
    pub(crate) fn map(layout: Layout, remainder: Remainder) -> io::Result<NonNull<u8>> {
        let (len, align) = mapped_len_and_align(layout);
        let huge_page_align = align.max(HUGE_PAGE_SIZE);
        // Over-map by `huge_page_align`, so the range can shift onto a huge page boundary.
        let map_len = len
            .checked_add(huge_page_align)
            .ok_or(io::ErrorKind::OutOfMemory)?;
        let map = map_anonymous(map_len, 0)?;
        let head_len = match remainder {
            // A rounded up `len` spans whole huge pages, so its start on a huge page boundary puts
            // its end on one too.
            Remainder::AtEnd | Remainder::RoundUp => map.align_offset(huge_page_align),
            // `len` is a multiple of `align`, which divides `huge_page_align`, so an end on a huge
            // page boundary keeps the start aligned.
            Remainder::AtStart => map.as_ptr().wrapping_add(len).align_offset(huge_page_align),
        };
        // SAFETY: the head is shorter than the extra `huge_page_align`, so the range stays inside
        // the fresh mapping.
        let start = unsafe { trim(map, map_len, head_len, len) }?;
        // SAFETY: the range is mapped and unused.
        let advised = unsafe { libc::madvise(start.as_ptr().cast(), len, libc::MADV_HUGEPAGE) };
        // The advice only affects how the kernel backs the range, so the memory stays valid in
        // base pages when it fails.
        if advised != 0 {
            static WARN_ONCE: Once = Once::new();
            let error = io::Error::last_os_error();
            WARN_ONCE.call_once(|| {
                log::warn!(
                    "large allocations use base pages, because madvise(MADV_HUGEPAGE) failed: \
                     {error}; the kernel likely lacks transparent huge page support \
                     (CONFIG_TRANSPARENT_HUGEPAGE), which slows access to them"
                );
            });
        }
        Ok(start)
    }

    /// Maps zeroed memory for `layout` from the reserved huge page pool.
    ///
    /// Requires a `layout` size in whole huge pages, and fails when the pool lacks free pages.
    pub(crate) fn map_reserved(layout: Layout) -> io::Result<NonNull<u8>> {
        let (len, align) = mapped_len_and_align(layout);
        debug_assert!(len.is_multiple_of(HUGE_PAGE_SIZE));
        // The kernel starts these mappings at a huge page boundary, so only larger alignments
        // need over-mapping.
        let map_len = len
            .checked_add(align.saturating_sub(HUGE_PAGE_SIZE))
            .ok_or(io::ErrorKind::OutOfMemory)?;
        // An explicit size keeps 2 MiB pages on hosts with another default huge page size.
        let map = map_anonymous(map_len, libc::MAP_HUGETLB | libc::MAP_HUGE_2MB)?;
        let head_len = map.align_offset(align);
        // SAFETY: the kernel aligns the mapping to a huge page, so the head is shorter than the
        // extra `align` and the range stays inside the fresh mapping.
        unsafe { trim(map, map_len, head_len, len) }
    }

    /// # Safety
    ///
    /// `ptr` must come from a [`map`] or [`map_reserved`] call with the same `layout`, and nothing
    /// may use the memory afterwards.
    pub(crate) unsafe fn unmap(ptr: NonNull<u8>, layout: Layout) {
        let (len, _) = mapped_len_and_align(layout);
        // SAFETY: the caller passes a live mapping of `len` bytes.
        unsafe { unmap_range_or_leak(ptr, len) };
    }

    fn mapped_len_and_align(layout: Layout) -> (usize, usize) {
        let align = layout.align().max(page_size());
        (layout.size().next_multiple_of(align), align)
    }

    fn map_anonymous(len: usize, flags: libc::c_int) -> io::Result<NonNull<u8>> {
        // SAFETY: an anonymous private mapping has no preconditions.
        let map = unsafe {
            libc::mmap(
                ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | flags,
                /* fd */ -1,
                /* offset */ 0,
            )
        };
        if ptr::eq(map, libc::MAP_FAILED) {
            return Err(io::Error::last_os_error());
        }
        NonNull::new(map.cast::<u8>()).ok_or(io::ErrorKind::OutOfMemory.into())
    }

    /// Returns a pointer `head_len` bytes into the `map_len` bytes mapped at `map`, after unmapping
    /// the parts before it and past its `kept_len` bytes.
    ///
    /// Fails when an unmap fails, after unmapping whatever is still mapped.
    ///
    /// # Safety
    ///
    /// `head_len + kept_len` must not exceed `map_len`, and nothing may reference the mapping.
    unsafe fn trim(
        map: NonNull<u8>,
        map_len: usize,
        head_len: usize,
        kept_len: usize,
    ) -> io::Result<NonNull<u8>> {
        let end_offset = head_len.strict_add(kept_len);
        let tail_len = map_len.strict_sub(end_offset);
        // SAFETY: the caller keeps the range inside the mapping.
        let (start, end) = unsafe { (map.add(head_len), map.add(end_offset)) };
        // SAFETY: the head and tail lie inside the mapping, and nothing references them.
        let (head_error, tail_error) = unsafe {
            (
                try_unmap_range(map, head_len).err(),
                try_unmap_range(end, tail_len).err(),
            )
        };
        let mapped_start = if head_error.is_some() { 0 } else { head_len };
        let mapped_end = if tail_error.is_some() {
            map_len
        } else {
            end_offset
        };
        let Some(error) = head_error.or(tail_error) else {
            return Ok(start);
        };
        // Unmap only the part still mapped, since another thread may already reuse the trimmed
        // addresses.
        // SAFETY: the part lies inside the mapping, and nothing references it.
        unsafe { unmap_range_or_leak(map.add(mapped_start), mapped_end.strict_sub(mapped_start)) };
        Err(error)
    }

    /// # Safety
    ///
    /// The range must lie inside a mapping, and nothing may reference it.
    unsafe fn try_unmap_range(ptr: NonNull<u8>, len: usize) -> io::Result<()> {
        // SAFETY: the caller passes an unreferenced part of a mapping.
        if len > 0 && unsafe { libc::munmap(ptr.as_ptr().cast(), len) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// # Safety
    ///
    /// The range must lie inside a mapping, and nothing may reference it.
    unsafe fn unmap_range_or_leak(ptr: NonNull<u8>, len: usize) {
        // SAFETY: the caller's guarantees carry over.
        if let Err(error) = unsafe { try_unmap_range(ptr, len) } {
            log::warn!("munmap failed, leaking {len} bytes: {error}");
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_map_places_huge_page_boundary() {
            for (align, remainder) in [
                (1, Remainder::AtEnd),
                (1, Remainder::AtStart),
                (2 * HUGE_PAGE_SIZE, Remainder::AtStart),
            ] {
                let layout = Layout::from_size_align(3 * HUGE_PAGE_SIZE + 4097, align).unwrap();
                let ptr = map(layout, remainder).unwrap();
                let (len, _) = mapped_len_and_align(layout);
                let start = ptr.addr().get();
                let boundary = match remainder {
                    Remainder::AtStart => start + len,
                    _ => start,
                };
                assert!(boundary.is_multiple_of(HUGE_PAGE_SIZE));
                assert!(start.is_multiple_of(align.max(page_size())));
                // SAFETY: the mapping holds `len` bytes and nothing else references it.
                let bytes = unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), len) };
                assert!(bytes.iter().all(|&b| b == 0));
                bytes.fill(1);
                // SAFETY: `ptr` comes from `map` with `layout`, and `bytes` is no longer used.
                unsafe { unmap(ptr, layout) };
            }
        }

        #[test]
        fn test_map_reserved_aligns_whole_huge_pages() {
            for align in [1, 2 * HUGE_PAGE_SIZE] {
                let layout = Layout::from_size_align(4 * HUGE_PAGE_SIZE, align).unwrap();
                // Hosts without a reserved pool only exercise the failure.
                let Ok(ptr) = map_reserved(layout) else {
                    continue;
                };
                assert!(ptr.addr().get().is_multiple_of(align.max(HUGE_PAGE_SIZE)));
                // SAFETY: the mapping holds the layout's bytes and nothing else references it.
                let bytes = unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), layout.size()) };
                assert!(bytes.iter().all(|&b| b == 0));
                bytes.fill(1);
                // SAFETY: `ptr` comes from `map_reserved` with `layout`, and `bytes` is no longer
                // used.
                unsafe { unmap(ptr, layout) };
            }
        }
    }
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn map(
    _layout: std::alloc::Layout,
    _remainder: crate::Remainder,
) -> std::io::Result<std::ptr::NonNull<u8>> {
    unreachable!("only Linux maps allocations directly")
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn map_reserved(_layout: std::alloc::Layout) -> std::io::Result<std::ptr::NonNull<u8>> {
    unreachable!("only Linux maps allocations directly")
}

#[cfg(not(target_os = "linux"))]
pub(crate) unsafe fn unmap(_ptr: std::ptr::NonNull<u8>, _layout: std::alloc::Layout) {
    unreachable!("only Linux maps allocations directly")
}
