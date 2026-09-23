//! Drives the accounts-lt-hash manager and its hasher workers on account
//! updates whose size mix matches steady mainnet-beta traffic.
//!
//! The benches in `solana-lattice-hash` time hashing alone. This one also runs
//! the dispatch path: queue, dedup, rayon spawn, per-worker accumulators and
//! freeze.

use {
    criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main},
    rand::{
        Rng, SeedableRng,
        distr::{Distribution, weighted::WeightedIndex},
        rngs::SmallRng,
    },
    solana_account::{Account, AccountSharedData},
    solana_lattice_hash::lt_hash::LtHash,
    solana_pubkey::Pubkey,
    solana_runtime::bank::accounts_lt_hash::{AccountsLtHashAsyncProgress, AccountsLtHashUpdate},
    std::{
        cell::LazyCell,
        hint,
        sync::{Arc, LazyLock},
        time::{Duration, Instant},
    },
};

/// `(account data length, measured weight per million updates)`. Each hashed
/// message adds 73 bytes of metadata to the account data. The 3762-byte entry
/// models vote accounts. Most slots update a few large accounts, so the last
/// entries carry a bump in weight.
const UPDATES_DATA_LEN_HISTOGRAM: &[(usize, u32)] = &[
    (0, 83_669),
    (82, 62_752),
    (165, 386_970),
    (200, 83_669),
    (500, 62_752),
    (900, 52_293),
    (1200, 25_328),
    (3762, 211_063),
    (10240, 26_079),
    (65_536, 2_405),
    (800_000, 823),
    (1_622_064, 2_176),
    (8_299_592, 21),
];

const UPDATES_WITH_PREV_PERCENT: u32 = 97;

/// Share of updates that rewrite an account already touched within this slot, to
/// match the measured dedup rate of about 8%.
const UPDATES_REWRITE_PERCENT: u32 = 8;

/// After dedup, about 3000 remain, the measured p50.
const NUM_UPDATES_PER_SLOT: usize = 3300;

/// Updates enqueued shortly before `finish()`, in bins of time before it:
/// `(bin end in us, measured per-slot update counts at p0, p10, ..., p100)`.
/// Each bin starts where the previous one ends.
const NUM_UPDATES_BEFORE_FINISH: [(f64, [u32; 11]); 3] = [
    (1000.0, [0, 1, 5, 10, 14, 19, 23, 29, 37, 65, 631]),
    (5000.0, [0, 27, 55, 75, 93, 113, 142, 188, 305, 714, 1805]),
    (20_000.0, [0, 0, 0, 0, 140, 241, 332, 440, 590, 875, 3653]),
];

/// New entries arrive for processing about every 5ms.
const ENTRY_POLL_INTERVAL_US: f64 = 5000.0;

/// Updates per `enqueue_for_dedup()` call, on average.
const MEAN_BATCH_LEN: f64 = 3.2;

/// Data lengths of freeze's own updates: the fee collector and the 1M-slot
/// `SlotHistory` bitmap. Freeze spawns them right before `finish()`.
const FREEZE_UPDATES_DATA_LENS: [usize; 2] = [0, 131_097];

/// Samples an index into `UPDATES_DATA_LEN_HISTOGRAM` by its weights.
static UPDATES_DATA_LEN_INDEX: LazyLock<WeightedIndex<u32>> = LazyLock::new(|| {
    WeightedIndex::new(UPDATES_DATA_LEN_HISTOGRAM.iter().map(|&(_, weight)| weight)).unwrap()
});

fn make_account(rng: &mut SmallRng, data_len: usize) -> AccountSharedData {
    AccountSharedData::from(Account {
        lamports: rng.random_range(1..=1_000_000),
        data: vec![0; data_len],
        owner: Pubkey::new_unique(),
        executable: false,
        rent_epoch: 0,
    })
}

/// Gives every update its own account data, so large accounts don't stay hot in
/// cache across updates.
fn create_updates_with_own_data(seed: u64, num_updates: usize) -> Vec<AccountsLtHashUpdate> {
    let rng = &mut SmallRng::seed_from_u64(seed);
    let mut touched = Vec::new();
    (0..num_updates)
        .map(|_| {
            let is_new_account =
                touched.is_empty() || !rng.random_ratio(UPDATES_REWRITE_PERCENT, 100);
            let index = if is_new_account {
                let data_len = UPDATES_DATA_LEN_HISTOGRAM[UPDATES_DATA_LEN_INDEX.sample(rng)].0;
                let prev = rng
                    .random_ratio(UPDATES_WITH_PREV_PERCENT, 100)
                    .then(|| make_account(rng, data_len));
                let index = touched.len();
                touched.push((Pubkey::new_unique(), data_len, prev));
                index
            } else {
                rng.random_range(0..touched.len())
            };
            let (address, data_len, last_version) = &mut touched[index];
            let curr = make_account(rng, *data_len);
            AccountsLtHashUpdate {
                address: *address,
                // Take the previous version from the last write to that address, as a
                // bank does.
                prev_account: last_version.replace(curr.clone()),
                curr_account: Some(curr),
            }
        })
        .collect()
}

fn enqueue_and_finish(
    progress: &Arc<AccountsLtHashAsyncProgress>,
    updates: Vec<AccountsLtHashUpdate>,
) -> LtHash {
    progress.enqueue_for_dedup(updates);
    let mut lt_hash = LtHash::identity();
    progress.finish(&mut lt_hash);
    lt_hash
}

/// Measures dispatch and hashing of whole slots: enqueue every update, then
/// freeze.
///
/// The sweep reaches ~10x a typical slot's updates, since a typical slot leaves
/// the pipeline mostly idle and only larger slots saturate it.
fn bench_slot_throughput(c: &mut Criterion) {
    const NUM_UPDATES_PER_SLOT_SWEEP: [usize; 4] = [512, NUM_UPDATES_PER_SLOT, 8192, 32768];
    let mut group = c.benchmark_group("accounts_lt_hash_slot");
    for num_updates in NUM_UPDATES_PER_SLOT_SWEEP {
        let updates = LazyCell::new(|| create_updates_with_own_data(7, num_updates));
        group.throughput(Throughput::Elements(num_updates as u64));
        group.bench_function(BenchmarkId::from_parameter(num_updates), |b| {
            b.iter_batched(
                || updates.to_vec(),
                |updates| {
                    enqueue_and_finish(&Arc::new(AccountsLtHashAsyncProgress::new()), updates)
                },
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

/// The freeze path alone, with the slot already hashed and only `num_pending`
/// updates left.
///
/// With no updates left, the time is the fixed cost of `finish()`. Each extra
/// update adds its own cost on top.
fn bench_freeze_pending(c: &mut Criterion) {
    let prefill = LazyCell::new(|| create_updates_with_own_data(11, NUM_UPDATES_PER_SLOT));
    let mut group = c.benchmark_group("accounts_lt_hash_freeze_pending");
    for &num_pending in &[0usize, 1, 16, 64, 256, 1024] {
        let pending_updates = LazyCell::new(|| create_updates_with_own_data(13, num_pending));
        group.bench_function(BenchmarkId::from_parameter(num_pending), |b| {
            b.iter_batched(
                || {
                    let progress = Arc::new(AccountsLtHashAsyncProgress::new());
                    // Prefill a regular slot's updates to warm up the pipeline, then let it
                    // go quiet, so the timed region sees only the pending updates and
                    // `finish()`.
                    progress.enqueue_for_dedup(prefill.iter().cloned());
                    progress.wait_for_pending_jobs();
                    (progress, pending_updates.to_vec())
                },
                |(progress, updates)| enqueue_and_finish(&progress, updates),
                BatchSize::PerIteration,
            )
        });
    }
    group.finish();
}

/// Simulates the last 20ms before a slot's freeze and returns how long
/// `finish()` takes.
///
/// Update batches arrive at times sampled from `NUM_UPDATES_BEFORE_FINISH`.
/// The end-of-slot signal comes at a random point within the entry poll
/// interval. Freeze then spawns its own updates and calls `finish()`, the only
/// timed step.
fn simulate_slot_end(
    rng: &mut SmallRng,
    shared_accounts_by_len: &[AccountSharedData],
    freeze_updates: &[AccountsLtHashUpdate],
) -> Duration {
    // Based on measured percentiles: pick a decile interval, then the bin's update
    // count uniformly within it.
    let sample_num_updates_in_bin = |rng: &mut SmallRng, counts_by_decile: &[u32; 11]| {
        let bounds = counts_by_decile
            .windows(2)
            .nth(rng.random_range(0..10))
            .unwrap();
        rng.random_range(bounds[0]..=bounds[1])
    };
    // Geometric with the measured mean, spread wide to cover various traffic shapes.
    // Regular traffic clusters closer to 2, since each vote transaction updates its
    // fee payer and vote account.
    let sample_batch_len = |rng: &mut SmallRng| -> usize {
        (1..)
            .find(|_| rng.random_bool(1.0 / MEAN_BATCH_LEN))
            .unwrap()
    };
    // BLAKE3's cost depends on input length, not content, so updates can share one
    // account per length.
    let sample_update = |rng: &mut SmallRng| {
        let account = &shared_accounts_by_len[UPDATES_DATA_LEN_INDEX.sample(rng)];
        AccountsLtHashUpdate {
            address: Pubkey::new_unique(),
            prev_account: rng
                .random_ratio(UPDATES_WITH_PREV_PERCENT, 100)
                .then(|| account.clone()),
            curr_account: Some(account.clone()),
        }
    };

    // Batches as `(start since the window opens, length)`, each within its bin.
    let window_us = NUM_UPDATES_BEFORE_FINISH.last().unwrap().0;
    let mut batches = Vec::new();
    let mut bin_start_us = 0.0;
    for (bin_end_us, counts_by_decile) in NUM_UPDATES_BEFORE_FINISH {
        let mut num_updates = sample_num_updates_in_bin(rng, &counts_by_decile) as usize;
        while num_updates > 0 {
            let batch_len = sample_batch_len(rng).min(num_updates);
            num_updates = num_updates.saturating_sub(batch_len);
            batches.push((
                window_us - rng.random_range(bin_start_us..bin_end_us),
                batch_len,
            ));
        }
        bin_start_us = bin_end_us;
    }
    batches.sort_by(|(a, _), (b, _)| a.total_cmp(b));
    let mut signal_us = Some(window_us - rng.random_range(0.0..ENTRY_POLL_INTERVAL_US));

    let progress = Arc::new(AccountsLtHashAsyncProgress::new());
    let window_start = Instant::now();
    let spin_until = |at_us: f64| {
        let at = Duration::from_secs_f64(at_us / 1e6);
        while window_start.elapsed() < at {
            hint::spin_loop();
        }
    };
    for (at_us, batch_len) in batches {
        if let Some(signal_us) = signal_us.take_if(|signal_us| *signal_us <= at_us) {
            spin_until(signal_us);
            progress.set_is_at_end_of_slot();
        }
        spin_until(at_us);
        progress.enqueue_for_dedup((0..batch_len).map(|_| sample_update(rng)));
    }
    spin_until(window_us);
    progress.spawn_deduped(freeze_updates.iter().cloned());
    let mut lt_hash = LtHash::identity();
    let start = Instant::now();
    progress.finish(&mut lt_hash);
    start.elapsed()
}

/// Measures how long `finish()` takes at the end of slot processing.
///
/// Prints the p10, p50, p90 and p99 over the slots criterion runs, to compare
/// with the measured `spin_us`.
fn bench_slot_freeze(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(17);
    let shared_accounts_by_len: Vec<_> = UPDATES_DATA_LEN_HISTOGRAM
        .iter()
        .map(|&(data_len, _)| make_account(&mut rng, data_len))
        .collect();
    let freeze_updates = FREEZE_UPDATES_DATA_LENS.map(|data_len| AccountsLtHashUpdate {
        address: Pubkey::new_unique(),
        prev_account: Some(make_account(&mut rng, data_len)),
        curr_account: Some(make_account(&mut rng, data_len)),
    });

    let mut durations = Vec::new();
    c.bench_function("accounts_lt_hash_slot_freeze", |b| {
        b.iter_custom(|iters| {
            (0..iters)
                .map(|_| simulate_slot_end(&mut rng, &shared_accounts_by_len, &freeze_updates))
                .inspect(|&duration| durations.push(duration))
                .sum()
        })
    });
    if !durations.is_empty() {
        durations.sort();
        let [p10, p50, p90, p99] = [0.1, 0.5, 0.9, 0.99]
            .map(|quantile| durations[(durations.len() as f64 * quantile) as usize].as_micros());
        eprintln!("slot freeze: p10={p10}us p50={p50}us p90={p90}us p99={p99}us\n");
    }
}

criterion_group!(
    benches,
    bench_slot_throughput,
    bench_freeze_pending,
    bench_slot_freeze
);
criterion_main!(benches);
