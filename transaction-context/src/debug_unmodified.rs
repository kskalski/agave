//! DEBUG instrumentation for `bank-accounts_lt_hash.mean_num_accounts_unmodified`.
//!
//! Quantifies *why* an account gets written back without its state actually changing. There are
//! two distinct causes, and this module exists to measure their relative weight:
//!
//! 1. **Round trip** - a field is changed and changed back within one transaction (e.g. a flash
//!    loan moving lamports out and back). Every individual write is a genuine change at the
//!    moment it happens, so no per-setter comparison can suppress it; only comparing the final
//!    state against the initial one can detect it.
//! 2. **No-op write** - the loader/CPI writes back byte-identical data in a single write. This
//!    one *is* suppressible at the write site by comparing first.
//!
//! Observation is kept strictly separate from suppression. By default a byte-identical write is
//! recorded and then still performed, so the touched flag is set exactly as in production and the
//! account reaches the detection in the runtime to be attributed. Suppression is a separate
//! opt-in (`AGAVE_SUPPRESS_NOOP_DATA_WRITES`) used to A/B the fix, because a build that suppresses
//! cannot measure its own baseline: the suppressed account may never be touched, never be stored,
//! and so never show up to be counted.
//!
//! Per-account change counts are thread-local because a batch is executed and committed on the
//! same thread; the runtime reads them during commit and then clears them for the next batch.

use {
    crate::{instruction_accounts::BorrowedInstructionAccount, transaction::TransactionContext},
    solana_instruction::error::InstructionError,
    solana_pubkey::Pubkey,
    std::{
        cell::RefCell,
        collections::HashMap,
        panic::Location,
        sync::{
            LazyLock,
            atomic::{AtomicU64, Ordering},
        },
    },
};

/// How many distinct code locations to remember per account. There are only a handful of
/// write-back call sites, so this never realistically overflows.
const MAX_TRACKED_LOCATIONS: usize = 6;

/// Counts of writes per originating code location.
///
/// The location comes from `#[track_caller]` on the setters, so it is the call site in the
/// loader or CPI code rather than somewhere inside `transaction-context`. This is what actually
/// distinguishes `deserialize_parameters()` from `update_callee_account()`; the instruction stack
/// height cannot, because both run while the *writing* instruction is current.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LocationCounts {
    entries: [Option<(&'static Location<'static>, u16)>; MAX_TRACKED_LOCATIONS],
    /// Writes whose location did not fit in `entries`.
    pub overflow: u16,
}

impl LocationCounts {
    fn record(&mut self, location: &'static Location<'static>) {
        for slot in self.entries.iter_mut() {
            match slot {
                Some((existing, count))
                    if existing.file() == location.file() && existing.line() == location.line() =>
                {
                    *count = count.saturating_add(1);
                    return;
                }
                Some(_) => continue,
                None => {
                    *slot = Some((location, 1));
                    return;
                }
            }
        }
        self.overflow = self.overflow.saturating_add(1);
    }

    pub fn iter(&self) -> impl Iterator<Item = (&'static Location<'static>, u16)> + '_ {
        self.entries.iter().flatten().copied()
    }

    pub fn is_empty(&self) -> bool {
        self.entries[0].is_none()
    }

    /// Renders as `serialization.rs:672x2,cpi.rs:1152x1`, dropping directories so the field stays
    /// short enough to keep one event on one line.
    pub fn render(&self) -> String {
        let mut rendered: Vec<String> = self
            .iter()
            .map(|(location, count)| {
                let file = location.file().rsplit('/').next().unwrap_or(location.file());
                format!("{file}:{}x{count}", location.line())
            })
            .collect();
        if self.overflow > 0 {
            rendered.push(format!("overflow x{}", self.overflow));
        }
        if rendered.is_empty() {
            "-".to_string()
        } else {
            rendered.join(",")
        }
    }
}

/// Whether the instrumentation is enabled, via the `AGAVE_DEBUG_UNMODIFIED_ACCOUNTS` env var.
///
/// Checked before any recording so the cost stays off by default.
pub fn enabled() -> bool {
    static ENABLED: LazyLock<bool> =
        LazyLock::new(|| std::env::var_os("AGAVE_DEBUG_UNMODIFIED_ACCOUNTS").is_some());
    *ENABLED
}

/// Whether to actually skip byte-identical data writes, via `AGAVE_SUPPRESS_NOOP_DATA_WRITES`.
///
/// Off by default: the point of the instrumentation is to measure production behaviour. Turn it
/// on only to compare a suppressing build against a baseline one.
pub fn suppress_noop_data_writes() -> bool {
    static SUPPRESS: LazyLock<bool> =
        LazyLock::new(|| std::env::var_os("AGAVE_SUPPRESS_NOOP_DATA_WRITES").is_some());
    *SUPPRESS
}

/// Whether a write site should compare before writing. Only then is the comparison worth its
/// cost: either we are measuring, or we are suppressing.
fn compare_before_write() -> bool {
    enabled() || suppress_noop_data_writes()
}

/// Where in a transaction a write happened.
///
/// Both write-back paths run while the *writing* program's instruction is still current:
/// `deserialize_parameters()` runs as an instruction returns but before it is popped, and
/// `update_callee_account()` runs during CPI setup before the callee is pushed. So the current
/// instruction context names the program that performed the write, not the account's owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WriteSite {
    /// Index of the instruction in the transaction's instruction trace, CPIs included.
    pub trace_index: u16,
    /// 1 for a top-level instruction, higher inside CPI.
    pub stack_height: u8,
    /// The program executing when the write happened.
    pub program: Pubkey,
}

/// Captures where execution currently is, or `None` when instrumentation is off.
pub fn current_write_site(transaction_context: &TransactionContext) -> Option<WriteSite> {
    if !enabled() {
        return None;
    }
    let instruction_context = transaction_context.get_current_instruction_context().ok()?;
    Some(WriteSite {
        trace_index: transaction_context.get_current_instruction_index().ok()? as u16,
        stack_height: instruction_context.get_stack_height() as u8,
        program: *instruction_context.get_program_key().ok()?,
    })
}

/// Number of *effective* changes to each field of one account during the current batch.
///
/// Only changes that actually set the touched flag are counted; setters that early-return
/// because the value is unchanged never reach here.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FieldChanges {
    pub lamports: u32,
    /// Data writes that reached `set_data_from_slice()` and friends, including no-op ones.
    pub data: u32,
    /// Subset of `data` that wrote byte-identical content.
    pub data_noop: u32,
    pub data_len: u32,
    pub owner: u32,
    /// Where the first and last write to this account happened. For a round trip these differ,
    /// and together they say which instruction changed the account and which changed it back.
    pub first_write: Option<WriteSite>,
    pub last_write: Option<WriteSite>,
    /// Which code locations performed the writes, and how many each did.
    pub locations: LocationCounts,
}

impl FieldChanges {
    /// Data writes that actually altered the bytes.
    ///
    /// Saturating because a suppressed no-op write is recorded without a matching `data` write.
    pub fn real_data_writes(&self) -> u32 {
        self.data.saturating_sub(self.data_noop)
    }

    fn note_site(&mut self, site: Option<WriteSite>) {
        if let Some(site) = site {
            self.first_write.get_or_insert(site);
            self.last_write = Some(site);
        }
    }
}

thread_local! {
    static CHANGES: RefCell<HashMap<Pubkey, FieldChanges>> = RefCell::new(HashMap::new());
}

fn record(
    address: &Pubkey,
    site: Option<WriteSite>,
    location: Option<&'static Location<'static>>,
    field: impl Fn(&mut FieldChanges),
) {
    if !enabled() {
        return;
    }
    CHANGES.with(|changes| {
        let mut changes = changes.borrow_mut();
        let entry = changes.entry(*address).or_default();
        field(entry);
        entry.note_site(site);
        if let Some(location) = location {
            entry.locations.record(location);
        }
    });
}

pub fn record_lamports_change(
    address: &Pubkey,
    site: Option<WriteSite>,
    location: &'static Location<'static>,
) {
    record(address, site, Some(location), |c| c.lamports = c.lamports.saturating_add(1));
}

pub fn record_data_change(
    address: &Pubkey,
    site: Option<WriteSite>,
    location: &'static Location<'static>,
) {
    record(address, site, Some(location), |c| c.data = c.data.saturating_add(1));
}

pub fn record_data_len_change(
    address: &Pubkey,
    site: Option<WriteSite>,
    location: &'static Location<'static>,
) {
    record(address, site, Some(location), |c| c.data_len = c.data_len.saturating_add(1));
}

pub fn record_owner_change(
    address: &Pubkey,
    site: Option<WriteSite>,
    location: &'static Location<'static>,
) {
    record(address, site, Some(location), |c| c.owner = c.owner.saturating_add(1));
}

fn record_noop_data_write(
    address: &Pubkey,
    site: Option<WriteSite>,
    location: &'static Location<'static>,
) {
    inc_noop_data_writes_seen();
    // Attributing the location here as well would double count: unless the write is suppressed,
    // set_data_from_slice_at() runs straight after and records the same location for the same
    // write. When it is suppressed there is no follow-up, so this is the only chance.
    let location = suppress_noop_data_writes().then_some(location);
    record(address, site, location, |c| {
        c.data_noop = c.data_noop.saturating_add(1)
    });
}

/// Writes `data` into `account`, observing whether the write is a no-op.
///
/// Byte-identical writes are counted and, unless suppression is enabled, still performed so the
/// touched flag matches production. This is the single place the loader and CPI write-back paths
/// should go through.
#[track_caller]
pub fn write_data_observed(
    account: &mut BorrowedInstructionAccount<'_, '_>,
    data: &[u8],
) -> Result<(), InstructionError> {
    // Forwarded explicitly rather than relying on #[track_caller] chaining, so the recorded
    // location is the loader/CPI call site and not this function.
    let location = Location::caller();
    if compare_before_write() && account.get_data() == data {
        let site = current_write_site(account.transaction_context);
        record_noop_data_write(account.get_key(), site, location);
        if suppress_noop_data_writes() {
            return Ok(());
        }
    }
    account.set_data_from_slice_at(data, location)
}

/// Returns the changes recorded for `address` so far in the current batch.
pub fn changes_for(address: &Pubkey) -> FieldChanges {
    if !enabled() {
        return FieldChanges::default();
    }
    CHANGES.with(|changes| changes.borrow().get(address).copied().unwrap_or_default())
}

/// Drops all per-account counts. Called by the runtime once a batch has been committed.
pub fn clear_batch() {
    if !enabled() {
        return;
    }
    CHANGES.with(|changes| changes.borrow_mut().clear());
}

/// Cause attributed to a single unmodified-account event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnmodifiedCause {
    /// Data was written more than once and ended up byte-identical. Irreducible at the write site.
    DataRoundTrip,
    /// Lamports were changed more than once and ended back at the original value. Irreducible.
    LamportsRoundTrip,
    /// Every write this account saw was byte-identical; suppressing them would avoid the touch.
    NoopDataWrite,
    /// Exactly one effective change was seen, yet the value is unchanged. Should be impossible,
    /// and would mean the previous version was loaded from the wrong slot.
    Inconsistent,
    /// Nothing was recorded, so some other path set the touched flag.
    Unattributed,
}

impl UnmodifiedCause {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DataRoundTrip => "data-roundtrip",
            Self::LamportsRoundTrip => "lamports-roundtrip",
            Self::NoopDataWrite => "noop-data-write",
            Self::Inconsistent => "inconsistent",
            Self::Unattributed => "unattributed",
        }
    }

    /// Whether suppressing no-op writes at the write site would have avoided this event.
    pub fn is_suppressible(&self) -> bool {
        matches!(self, Self::NoopDataWrite)
    }
}

/// Classifies an unmodified-account event from the changes recorded for it.
///
/// Round trips are tested first: if an account both round-tripped and saw a no-op write, the
/// round trip alone would still have set the touched flag, so suppressing the no-op write would
/// not have helped. Only when *every* recorded write was a no-op is the event suppressible.
pub fn classify(changes: &FieldChanges) -> UnmodifiedCause {
    if changes.real_data_writes() >= 2 || changes.data_len >= 2 {
        UnmodifiedCause::DataRoundTrip
    } else if changes.lamports >= 2 {
        UnmodifiedCause::LamportsRoundTrip
    } else if changes.data_noop >= 1 {
        UnmodifiedCause::NoopDataWrite
    } else if changes.real_data_writes() == 1 || changes.lamports == 1 || changes.data_len == 1 {
        UnmodifiedCause::Inconsistent
    } else {
        UnmodifiedCause::Unattributed
    }
}

macro_rules! counters {
    ($($name:ident => ($inc:ident, $get:ident)),* $(,)?) => {
        $(
            static $name: AtomicU64 = AtomicU64::new(0);
            pub fn $inc() { $name.fetch_add(1, Ordering::Relaxed); }
            pub fn $get() -> u64 { $name.load(Ordering::Relaxed) }
        )*
    };
}

counters! {
    // Unmodified-account events, split by attributed cause.
    DATA_ROUNDTRIP => (inc_data_roundtrip, data_roundtrip),
    LAMPORTS_ROUNDTRIP => (inc_lamports_roundtrip, lamports_roundtrip),
    NOOP_DATA_WRITE => (inc_noop_data_write, noop_data_write),
    INCONSISTENT => (inc_inconsistent, inconsistent),
    UNATTRIBUTED => (inc_unattributed, unattributed),
    // Every byte-identical data write seen, whether or not it was suppressed and whether or not
    // the account ended up unmodified. An upper bound on what suppression could ever avoid.
    NOOP_DATA_WRITES_SEEN => (inc_noop_data_writes_seen, noop_data_writes_seen),
}

/// Tallies one unmodified-account event under its attributed cause.
pub fn count(cause: UnmodifiedCause) {
    match cause {
        UnmodifiedCause::DataRoundTrip => inc_data_roundtrip(),
        UnmodifiedCause::LamportsRoundTrip => inc_lamports_roundtrip(),
        UnmodifiedCause::NoopDataWrite => inc_noop_data_write(),
        UnmodifiedCause::Inconsistent => inc_inconsistent(),
        UnmodifiedCause::Unattributed => inc_unattributed(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn changes(lamports: u32, data: u32, data_noop: u32, data_len: u32) -> FieldChanges {
        FieldChanges {
            lamports,
            data,
            data_noop,
            data_len,
            ..Default::default()
        }
    }

    #[test]
    fn test_classify_round_trips() {
        // two writes that each really changed the bytes
        assert_eq!(
            classify(&changes(0, 2, 0, 0)),
            UnmodifiedCause::DataRoundTrip
        );
        assert_eq!(
            classify(&changes(0, 0, 0, 2)),
            UnmodifiedCause::DataRoundTrip
        );
        assert_eq!(
            classify(&changes(2, 0, 0, 0)),
            UnmodifiedCause::LamportsRoundTrip
        );
    }

    #[test]
    fn test_classify_noop_write() {
        // observed but not suppressed: the write still happened, so `data` counts it too
        assert_eq!(
            classify(&changes(0, 1, 1, 0)),
            UnmodifiedCause::NoopDataWrite
        );
        // suppressed: no matching `data` write was recorded
        assert_eq!(
            classify(&changes(0, 0, 1, 0)),
            UnmodifiedCause::NoopDataWrite
        );
    }

    #[test]
    fn test_classify_prefers_irreducible_cause() {
        // a lamports round trip would set the touched flag regardless of the no-op write, so
        // suppressing the no-op write would not have avoided this event
        assert_eq!(
            classify(&changes(2, 1, 1, 0)),
            UnmodifiedCause::LamportsRoundTrip
        );
        assert!(!UnmodifiedCause::LamportsRoundTrip.is_suppressible());
        assert!(UnmodifiedCause::NoopDataWrite.is_suppressible());
    }

    #[test]
    fn test_location_counts_dedup_and_render() {
        let mut counts = LocationCounts::default();
        // two writes from one site, one from another
        let a = Location::caller();
        counts.record(a);
        counts.record(a);
        assert_eq!(counts.iter().count(), 1);
        assert_eq!(counts.iter().next().unwrap().1, 2);
        // rendering keeps only the file name, not the directory
        let rendered = counts.render();
        assert!(!rendered.contains('/'), "{rendered}");
        assert!(rendered.ends_with("x2"), "{rendered}");
    }

    #[test]
    fn test_location_counts_empty_renders_dash() {
        assert_eq!(LocationCounts::default().render(), "-");
        assert!(LocationCounts::default().is_empty());
    }

    #[test]
    fn test_note_site_keeps_first_and_last() {
        let site = |trace_index| WriteSite {
            trace_index,
            stack_height: 1,
            program: Pubkey::new_from_array([7; 32]),
        };
        let mut c = FieldChanges::default();
        c.note_site(Some(site(0)));
        c.note_site(Some(site(3)));
        c.note_site(None); // instrumentation off for this write, must not clobber
        assert_eq!(c.first_write.map(|s| s.trace_index), Some(0));
        assert_eq!(c.last_write.map(|s| s.trace_index), Some(3));
    }

    #[test]
    fn test_classify_degenerate() {
        assert_eq!(classify(&changes(0, 0, 0, 0)), UnmodifiedCause::Unattributed);
        assert_eq!(classify(&changes(1, 0, 0, 0)), UnmodifiedCause::Inconsistent);
        assert_eq!(classify(&changes(0, 1, 0, 0)), UnmodifiedCause::Inconsistent);
    }

    #[test]
    fn test_real_data_writes_saturates() {
        assert_eq!(changes(0, 0, 1, 0).real_data_writes(), 0);
        assert_eq!(changes(0, 3, 1, 0).real_data_writes(), 2);
    }
}
