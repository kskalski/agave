//! Drives the accounts-lt-hash manager and its hasher workers on a stream of
//! account updates whose size mix matches steady mainnet-beta traffic.
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

/// Share of updates carrying a previous version.
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

/// One account per `UPDATES_DATA_LEN_HISTOGRAM` entry. Hashing cost does not
/// depend on the content, so sampled updates share their data.
fn make_accounts_by_histogram_len(rng: &mut SmallRng) -> Vec<AccountSharedData> {
    UPDATES_DATA_LEN_HISTOGRAM
        .iter()
        .map(|&(data_len, _)| make_account(rng, data_len))
        .collect()
}

fn sample_update(
    rng: &mut SmallRng,
    accounts_by_len: &[AccountSharedData],
) -> AccountsLtHashUpdate {
    let account = &accounts_by_len[UPDATES_DATA_LEN_INDEX.sample(rng)];
    AccountsLtHashUpdate {
        address: Pubkey::new_unique(),
        prev_account: (rng.random_range(0..100) < UPDATES_WITH_PREV_PERCENT)
            .then(|| account.clone()),
        curr_account: Some(account.clone()),
    }
}

/// Samples a count from `counts_by_decile` (p0, p10, ..., p100): picks a decile
/// interval, then a count within it.
fn sample_count(rng: &mut SmallRng, counts_by_decile: &[u32; 11]) -> u32 {
    let bounds = counts_by_decile
        .windows(2)
        .nth(rng.random_range(0..10))
        .unwrap();
    rng.random_range(bounds[0]..=bounds[1])
}

/// Draws a batch length from a geometric distribution with mean `MEAN_BATCH_LEN`.
fn sample_batch_len(rng: &mut SmallRng) -> usize {
    (1..)
        .find(|_| rng.random_bool(1.0 / MEAN_BATCH_LEN))
        .unwrap()
}

/// Builds one slot's updates.
///
/// A rewrite takes its previous version from the last write to that address, as a
/// bank does.
fn make_stream(seed: u64, num_updates: usize) -> Vec<AccountsLtHashUpdate> {
    let rng = &mut SmallRng::seed_from_u64(seed);
    let mut touched = Vec::new();
    (0..num_updates)
        .map(|_| {
            let is_new_account =
                touched.is_empty() || rng.random_range(0..100) >= UPDATES_REWRITE_PERCENT;
            let index = if is_new_account {
                let data_len = UPDATES_DATA_LEN_HISTOGRAM[UPDATES_DATA_LEN_INDEX.sample(rng)].0;
                let prev = (rng.random_range(0..100) < UPDATES_WITH_PREV_PERCENT)
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
                prev_account: last_version.replace(curr.clone()),
                curr_account: Some(curr),
            }
        })
        .collect()
}

/// Number of hashed messages in `stream`: one per prev or curr account.
fn num_hashed_messages(stream: &[AccountsLtHashUpdate]) -> u64 {
    stream
        .iter()
        .flat_map(|update| [&update.prev_account, &update.curr_account])
        .flatten()
        .count() as u64
}

/// Measures dispatch and hashing of whole slots: enqueue every update, then
/// freeze.
///
/// The sweep reaches ~10x a typical slot's updates, since a typical slot leaves
/// the pipeline mostly idle and only larger slots saturate it.
/// `AGAVE_LT_HASH_BENCH_SLOT_SIZES` overrides the sweep.
fn bench_slot_throughput(c: &mut Criterion) {
    const NUM_UPDATES_PER_SLOT_SWEEP: &[usize] = &[512, 2048, 8192, 32768];
    let num_updates_per_slot = match std::env::var("AGAVE_LT_HASH_BENCH_SLOT_SIZES") {
        Ok(spec) => spec
            .split(',')
            .map(|size| size.trim().parse().expect("slot size must be a number"))
            .collect(),
        Err(_) => NUM_UPDATES_PER_SLOT_SWEEP.to_vec(),
    };
    let mut group = c.benchmark_group("accounts_lt_hash_slot");
    for num_updates in num_updates_per_slot {
        let stream = LazyCell::new(|| make_stream(7, num_updates));
        group.throughput(Throughput::Elements(num_updates as u64));
        group.bench_function(BenchmarkId::from_parameter(num_updates), |b| {
            b.iter_batched(
                || stream.to_vec(),
                |stream| {
                    let progress = Arc::new(AccountsLtHashAsyncProgress::new());
                    progress.enqueue_for_dedup(stream);
                    let mut lt_hash = LtHash::identity();
                    progress.finish(&mut lt_hash);
                    lt_hash
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
    let prefill = LazyCell::new(|| make_stream(11, NUM_UPDATES_PER_SLOT));
    let mut group = c.benchmark_group("accounts_lt_hash_freeze_pending");
    for &num_pending in &[0usize, 1, 16, 64, 256, 1024] {
        let pending = LazyCell::new(|| make_stream(13, num_pending));
        group.bench_function(BenchmarkId::from_parameter(num_pending), |b| {
            b.iter_batched(
                || {
                    let progress = Arc::new(AccountsLtHashAsyncProgress::new());
                    // Prefill a regular slot's updates to warm up the pipeline, then let it
                    // go quiet, so the timed region sees only the pending updates and
                    // `finish()`.
                    progress.enqueue_for_dedup(prefill.iter().cloned());
                    while progress.num_pending() > 0 {
                        std::thread::yield_now();
                    }
                    (progress, pending.to_vec())
                },
                |(progress, pending)| {
                    progress.enqueue_for_dedup(pending);
                    let mut lt_hash = LtHash::identity();
                    progress.finish(&mut lt_hash);
                    lt_hash
                },
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
    accounts_by_len: &[AccountSharedData],
    freeze_updates: &[AccountsLtHashUpdate],
) -> Duration {
    let mut before_finish_us = Vec::new();
    let mut bin_start_us = 0.0;
    for (bin_end_us, counts_by_decile) in NUM_UPDATES_BEFORE_FINISH {
        for _ in 0..sample_count(rng, &counts_by_decile) {
            before_finish_us.push(rng.random_range(bin_start_us..bin_end_us));
        }
        bin_start_us = bin_end_us;
    }
    before_finish_us.sort_by(|a, b| b.total_cmp(a));
    let mut signal_before_us = Some(rng.random_range(0.0..ENTRY_POLL_INTERVAL_US));

    let progress = Arc::new(AccountsLtHashAsyncProgress::new());
    let finish_at = Instant::now()
        .checked_add(Duration::from_secs_f64(bin_start_us / 1e6))
        .unwrap();
    let wait_until_before = |before_us: f64| {
        let deadline = finish_at
            .checked_sub(Duration::from_secs_f64(before_us / 1e6))
            .unwrap();
        while Instant::now() < deadline {
            hint::spin_loop();
        }
    };
    let mut before_finish_us = before_finish_us.into_iter();
    while let Some(before_us) = before_finish_us.next() {
        if let Some(signal_us) = signal_before_us.filter(|&signal_us| signal_us > before_us) {
            wait_until_before(signal_us);
            progress.set_is_at_end_of_slot();
            signal_before_us = None;
        }
        wait_until_before(before_us);
        let batch_len = sample_batch_len(rng);
        for _ in 1..batch_len {
            before_finish_us.next();
        }
        progress.enqueue_for_dedup((0..batch_len).map(|_| sample_update(rng, accounts_by_len)));
    }
    wait_until_before(0.0);
    progress.spawn_deduped(freeze_updates.iter().cloned());
    let mut lt_hash = LtHash::identity();
    let start = Instant::now();
    progress.finish(&mut lt_hash);
    start.elapsed()
}

/// Measures how long `finish()` takes at the end of slot processing.
///
/// Prints the p10, p50, p90 and p99 of 1000 sampled slots, to compare with the
/// measured `spin_us`.
fn bench_slot_freeze(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(17);
    let accounts_by_len = make_accounts_by_histogram_len(&mut rng);
    let freeze_updates = FREEZE_UPDATES_DATA_LENS.map(|data_len| AccountsLtHashUpdate {
        address: Pubkey::new_unique(),
        prev_account: Some(make_account(&mut rng, data_len)),
        curr_account: Some(make_account(&mut rng, data_len)),
    });

    let mut samples: Vec<_> = (0..1000)
        .map(|_| simulate_slot_end(&mut rng, &accounts_by_len, &freeze_updates))
        .collect();
    samples.sort();
    let [p10, p50, p90, p99] = [0.1, 0.5, 0.9, 0.99]
        .map(|quantile| samples[(samples.len() as f64 * quantile) as usize].as_micros());
    eprintln!("slot freeze: p10={p10}us p50={p50}us p90={p90}us p99={p99}us\n");

    c.bench_function("accounts_lt_hash_slot_freeze", |b| {
        b.iter_custom(|iters| {
            (0..iters)
                .map(|_| simulate_slot_end(&mut rng, &accounts_by_len, &freeze_updates))
                .sum()
        })
    });
}

fn bench_all(c: &mut Criterion) {
    let stream = make_stream(7, NUM_UPDATES_PER_SLOT);
    eprintln!(
        "\naccounts_lt_hash bench stream: {} updates, {} hashed messages\n",
        stream.len(),
        num_hashed_messages(&stream),
    );
    bench_slot_throughput(c);
    bench_freeze_pending(c);
    bench_slot_freeze(c);
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
