use {
    super::{AtomicRefCount, DiskIndexValue, IndexValue, RefCount, SlotList},
    crate::{
        bucket_map_holder::{Age, AtomicAge, BucketMapHolder},
        is_zero_lamport::IsZeroLamport,
    },
    solana_clock::Slot,
    std::{
        fmt::Debug,
        ops::{Deref, DerefMut},
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc, RwLock, RwLockReadGuard,
        },
    },
};

#[derive(Debug)]
pub enum AccountMapEntryKind<T> {
    Regular((Slot, T)),
    Irregular(Box<IrregularAccountMapEntry<T>>),
}

/// one entry in the in-mem accounts index
/// Represents the value for an account key in the in-memory accounts index
pub struct AccountMapEntry<T>(RwLock<AccountMapEntryKind<T>>);

impl<T: IndexValue> AccountMapEntry<T> {
    pub fn new(slot_list: SlotList<T>, ref_count: RefCount, meta: AccountMapEntryMeta) -> Self {
        let kind = if slot_list.len() == 1 && ref_count == 1 && !meta.dirty.load(Ordering::Relaxed)
        {
            AccountMapEntryKind::Regular(slot_list[0])
        } else {
            AccountMapEntryKind::Irregular(Box::new(IrregularAccountMapEntry {
                slot_list,
                ref_count: AtomicRefCount::new(ref_count),
                meta,
            }))
        };
        Self(RwLock::new(kind))
    }

    #[cfg(test)]
    pub(super) fn empty_for_tests() -> Self {
        Self(RwLock::new(AccountMapEntryKind::Irregular(Box::new(
            IrregularAccountMapEntry::default(),
        ))))
    }

    pub fn entry_view(&self) -> RwLockReadGuard<AccountMapEntryKind<T>> {
        self.0.read().unwrap()
    }

    pub fn make_irregular(
        &self,
    ) -> impl DerefMut<Target = IrregularAccountMapEntry<T>> + use<'_, T> {
        let mut kind = self.0.write().unwrap();
        if let AccountMapEntryKind::Regular(entry) = &mut *kind {
            *kind = AccountMapEntryKind::Irregular(Box::new(IrregularAccountMapEntry {
                slot_list: SlotList::from([*entry]),
                ref_count: AtomicRefCount::new(1),
                meta: AccountMapEntryMeta::default(),
            }));
        }

        WriteGuardMap(
            kind,
            |kind| match kind {
                AccountMapEntryKind::Irregular(irregular) => irregular.as_ref(),
                _ => unreachable!(),
            },
            |kind| match kind {
                AccountMapEntryKind::Irregular(irregular) => irregular.as_mut(),
                _ => unreachable!(),
            },
        )
    }

    pub fn ref_count(&self) -> RefCount {
        self.entry_view().ref_count()
    }

    pub fn dirty(&self) -> bool {
        self.entry_view().dirty()
    }

    pub fn age(&self) -> Age {
        self.entry_view().age()
    }

    pub fn set_age(&self, value: Age) {
        self.make_irregular().set_age(value);
    }

    /// set age to 'next_age' if 'self.age' is 'expected_age'
    pub fn try_exchange_age(&self, next_age: Age, expected_age: Age) {
        let _ = self.make_irregular().meta.age.compare_exchange(
            expected_age,
            next_age,
            Ordering::AcqRel,
            Ordering::Relaxed,
        );
    }
}

impl<T: Debug> Debug for AccountMapEntry<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("AccountMapEntry").field(&self.0).finish()
    }
}

#[derive(Debug, Default)]
pub struct IrregularAccountMapEntry<T> {
    /// number of alive slots that contain >= 1 instances of account data for this pubkey
    /// where alive represents a slot that has not yet been removed by clean via AccountsDB::clean_stored_dead_slots() for containing no up to date account information
    ref_count: AtomicRefCount,
    /// list of slots in which this pubkey was updated
    /// Note that 'clean' removes outdated entries (ie. older roots) from this slot_list
    /// purge_slot() also removes non-rooted slots from this list
    pub slot_list: SlotList<T>,
    /// synchronization metadata for in-memory state since last flush to disk accounts index
    pub meta: AccountMapEntryMeta,
}

impl<T: Debug> IrregularAccountMapEntry<T> {
    pub fn from_count(ref_count: u32) -> Self {
        Self {
            ref_count: AtomicRefCount::new(ref_count),
            slot_list: SlotList::default(),
            meta: AccountMapEntryMeta::default(),
        }
    }

    pub fn ref_count(&self) -> RefCount {
        self.ref_count.load(Ordering::Acquire)
    }

    pub fn age(&self) -> Age {
        self.meta.age.load(Ordering::Acquire)
    }

    pub fn set_age(&self, value: Age) {
        self.meta.age.store(value, Ordering::Release)
    }

    pub fn dirty(&self) -> bool {
        self.meta.dirty.load(Ordering::Acquire)
    }

    pub fn set_dirty(&self, value: bool) {
        self.meta.dirty.store(value, Ordering::Release)
    }

    pub fn clear_dirty(&self) -> bool {
        self.meta
            .dirty
            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
    }

    pub fn addref(&self) {
        let previous = self.ref_count.fetch_add(1, Ordering::Release);
        // ensure ref count does not overflow
        assert_ne!(previous, RefCount::MAX);
        self.set_dirty(true);
    }

    /// decrement the ref count by one
    /// return the refcount prior to subtracting 1
    /// 0 indicates an under refcounting error in the system.
    pub fn unref(&self) -> RefCount {
        self.unref_by_count(1)
    }

    /// decrement the ref count by the passed in amount
    /// return the refcount prior to the ref count change
    pub fn unref_by_count(&self, count: RefCount) -> RefCount {
        let previous = self.ref_count.fetch_sub(count, Ordering::Release);
        self.set_dirty(true);
        assert!(
            previous >= count,
            "decremented ref count below zero: {self:?}"
        );
        previous
    }
}

struct WriteGuardMap<G, S, T, FR, F>(G, FR, F)
where
    G: DerefMut<Target = S>,
    for<'b> FR: Fn(&'b S) -> &'b T,
    for<'b> F: Fn(&'b mut S) -> &'b mut T;

impl<G, S, T, FR, F> Deref for WriteGuardMap<G, S, T, FR, F>
where
    G: DerefMut<Target = S>,
    for<'b> FR: Fn(&'b S) -> &'b T,
    for<'b> F: Fn(&'b mut S) -> &'b mut T,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.1(&*self.0)
    }
}

impl<G, S, T, FR, F> DerefMut for WriteGuardMap<G, S, T, FR, F>
where
    G: DerefMut<Target = S>,
    for<'b> FR: Fn(&'b S) -> &'b T,
    for<'b> F: Fn(&'b mut S) -> &'b mut T,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.2(&mut *self.0)
    }
}

impl<T: IndexValue> AccountMapEntryKind<T> {
    pub fn ref_count(&self) -> RefCount {
        match self {
            Self::Regular(_) => 1,
            Self::Irregular(irregular) => irregular.ref_count(),
        }
    }

    pub fn dirty(&self) -> bool {
        match self {
            AccountMapEntryKind::Regular(_) => false,
            AccountMapEntryKind::Irregular(irregular) => irregular.dirty(),
        }
    }

    pub fn age(&self) -> Age {
        match self {
            AccountMapEntryKind::Regular(_) => 0,
            AccountMapEntryKind::Irregular(irregular) => irregular.age(),
        }
    }

    pub fn is_zero_lamport(&self) -> bool {
        match self {
            AccountMapEntryKind::Regular((_, entry)) => entry.is_zero_lamport(),
            AccountMapEntryKind::Irregular(irregular) => irregular.slot_list[0].1.is_zero_lamport(),
        }
    }

    pub fn slot_list(&self) -> &[(Slot, T)] {
        match self {
            AccountMapEntryKind::Regular(entry) => std::slice::from_ref(entry),
            AccountMapEntryKind::Irregular(irregular) => irregular.slot_list.as_slice(),
        }
    }

    pub fn slot_list_len(&self) -> usize {
        match self {
            Self::Regular(_) => 1,
            Self::Irregular(irregular) => irregular.slot_list.len(),
        }
    }
}

/// data per entry in in-mem accounts index
/// used to keep track of consistency with disk index
#[derive(Debug, Default)]
pub struct AccountMapEntryMeta {
    /// true if entry in in-mem idx has changes and needs to be written to disk
    pub dirty: AtomicBool,
    /// 'age' at which this entry should be purged from the cache (implements lru)
    pub age: AtomicAge,
}

impl AccountMapEntryMeta {
    pub fn new_dirty<T: IndexValue, U: DiskIndexValue + From<T> + Into<T>>(
        storage: &BucketMapHolder<T, U>,
        is_cached: bool,
    ) -> Self {
        AccountMapEntryMeta {
            dirty: AtomicBool::new(true),
            age: AtomicAge::new(storage.future_age_to_flush(is_cached)),
        }
    }
    pub fn new_clean<T: IndexValue, U: DiskIndexValue + From<T> + Into<T>>(
        storage: &BucketMapHolder<T, U>,
    ) -> Self {
        AccountMapEntryMeta {
            dirty: AtomicBool::new(false),
            age: AtomicAge::new(storage.future_age_to_flush(false)),
        }
    }
}

/// can be used to pre-allocate structures for insertion into accounts index outside of lock
pub enum PreAllocatedAccountMapEntry<T: IndexValue> {
    Entry(Arc<AccountMapEntry<T>>),
    Raw((Slot, T)),
}

impl<T: IndexValue> IsZeroLamport for PreAllocatedAccountMapEntry<T> {
    fn is_zero_lamport(&self) -> bool {
        match self {
            PreAllocatedAccountMapEntry::Entry(entry) => entry.entry_view().is_zero_lamport(),
            PreAllocatedAccountMapEntry::Raw(raw) => raw.1.is_zero_lamport(),
        }
    }
}

impl<T: IndexValue> From<PreAllocatedAccountMapEntry<T>> for (Slot, T) {
    fn from(source: PreAllocatedAccountMapEntry<T>) -> (Slot, T) {
        match source {
            PreAllocatedAccountMapEntry::Entry(entry) => entry.entry_view().slot_list()[0],
            PreAllocatedAccountMapEntry::Raw(raw) => raw,
        }
    }
}

impl<T: IndexValue> PreAllocatedAccountMapEntry<T> {
    /// create an entry that is equivalent to this process:
    /// 1. new empty (refcount=0, slot_list={})
    /// 2. update(slot, account_info)
    ///
    /// This code is called when the first entry [ie. (slot,account_info)] for a pubkey is inserted into the index.
    pub fn new<U: DiskIndexValue + From<T> + Into<T>>(
        slot: Slot,
        account_info: T,
        storage: &BucketMapHolder<T, U>,
        store_raw: bool,
    ) -> PreAllocatedAccountMapEntry<T> {
        if store_raw {
            Self::Raw((slot, account_info))
        } else {
            Self::Entry(Self::allocate(slot, account_info, storage))
        }
    }

    fn allocate<U: DiskIndexValue + From<T> + Into<T>>(
        slot: Slot,
        account_info: T,
        storage: &BucketMapHolder<T, U>,
    ) -> Arc<AccountMapEntry<T>> {
        let is_cached = account_info.is_cached();
        let ref_count = RefCount::from(!is_cached);
        let meta = AccountMapEntryMeta::new_dirty(storage, is_cached);
        Arc::new(AccountMapEntry::new(
            SlotList::from([(slot, account_info)]),
            ref_count,
            meta,
        ))
    }

    pub fn into_account_map_entry<U: DiskIndexValue + From<T> + Into<T>>(
        self,
        storage: &BucketMapHolder<T, U>,
    ) -> Arc<AccountMapEntry<T>> {
        match self {
            Self::Entry(entry) => entry,
            Self::Raw((slot, account_info)) => Self::allocate(slot, account_info, storage),
        }
    }
}
