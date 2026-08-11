#!/usr/bin/env python3
"""Drill into unmodified-account events for one owner program.

Answers "which accounts is this program writing without changing them, and what are those
accounts" using only a log captured with AGAVE_DEBUG_UNMODIFIED_ACCOUNTS=1. No RPC, no chain
access: every column is derived from the log line itself.

Use `scripts/unmodified-accounts-stats.py --by-owner` first to rank the programs, then this to
drill into one of them.

Pre-filter with grep, it matters on multi-GB logs:

    grep 'owner=pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA' validator.log \\
      | scripts/unmodified-accounts-drill.py --cause noop-data-write

    scripts/unmodified-accounts-drill.py validator.log --owner pAMMBay... --addresses out.txt

What the derived columns mean:

  data_len    Distinct sizes usually map one-to-one onto account types, since a given struct
              serialises to a fixed length. Treat each size as a candidate account type.
  rent_min    The rent-exempt minimum for that size, (128 + data_len) * 3480 * 2. An account
              sitting exactly at it holds no working balance, which is the signature of a
              pure-state PDA (config, pool state). An account well above it is holding SOL,
              i.e. a vault or a wallet, whose lamports move for real reasons.
  writes      changes_data counts every write-back, changes_data_noop the byte-identical subset,
              so real writes are the difference. More than one write per event means the account
              was written back several times in the same transaction, typically once on a CPI
              return and once when the top-level instruction returns.
  wasted      Each no-op write on a still-shared buffer costs an allocation plus a memcpy of
              data_len bytes (AccountSharedData::set_data_from_slice falls back to to_vec() when
              the Arc is shared). Bytes, not events, are what the fix actually saves.
  write_locs  The source locations that performed the writes, as file:line xN. This is the
              authoritative answer for which code path wrote: stack height cannot tell
              deserialize_parameters() apart from update_callee_account(), since both run while
              the writing instruction is still current.
  write_*     Where in the transaction the write came from. The program is whoever was
              executing, not necessarily the account owner: a DEX CPI-ing into Token writes a
              Token-owned account while the DEX executes. write_ix and write_ix_last differ when
              more than one instruction wrote the account. Note write_ix is drawn from a
              different counter for top-level and CPI instructions, so values are only
              comparable within the same stack height.
"""

import argparse
import sys
from collections import Counter, defaultdict

LINE = "unmodified account written in lt_hash update:"

# solana_rent: (ACCOUNT_STORAGE_OVERHEAD + bytes) * lamports_per_byte_year * exemption_threshold
ACCOUNT_STORAGE_OVERHEAD = 128
DEFAULT_LAMPORTS_PER_BYTE_YEAR = 3480
DEFAULT_EXEMPTION_THRESHOLD = 2

INT_FIELDS = frozenset((
    "slot", "lamports", "data_len", "num_occurrences", "batch_len",
    "changes_lamports", "changes_data", "changes_data_noop", "changes_data_len", "changes_owner",
))


def rent_exempt_minimum(data_len):
    return ((ACCOUNT_STORAGE_OVERHEAD + data_len)
            * DEFAULT_LAMPORTS_PER_BYTE_YEAR
            * DEFAULT_EXEMPTION_THRESHOLD)


def parse_line(line):
    """Splits `k=v` tokens by hand; measurably faster than a regex over tens of millions of lines."""
    event = {}
    for token in line[line.index(LINE) + len(LINE):].split():
        key, sep, value = token.partition("=")
        if not sep:
            continue
        if key in INT_FIELDS:
            try:
                value = int(value)
            except ValueError:
                continue
        event[key] = value
    return event


def pct(n, total):
    return f"{100.0 * n / total:5.1f}%" if total else "    -"


def human_bytes(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} TiB"


def spread(values):
    values = sorted(values)
    if not values:
        return "n/a"
    n = len(values)
    return (f"mean {sum(values) / n:8.1f}  median {values[n // 2]:>8}  "
            f"p95 {values[min(n - 1, int(0.95 * n))]:>8}  max {values[-1]:>8}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", type=argparse.FileType("r", errors="replace"),
                    default=sys.stdin, help="validator log (default: stdin)")
    ap.add_argument("--owner", help="only events for accounts owned by this program")
    ap.add_argument("--cause", help="only this cause, e.g. noop-data-write")
    ap.add_argument("--top", type=int, default=20, help="rows in the tables")
    ap.add_argument("--addresses", help="write the distinct addresses, most frequent first")
    args = ap.parse_args()

    by_address = Counter()
    by_data_len = Counter()
    len_addresses = defaultdict(set)
    len_at_rent_min = Counter()
    address_meta = {}
    per_slot = Counter()
    causes = Counter()
    batch_lens = []
    writes_per_event = Counter()
    by_write_program = defaultdict(Counter)
    by_write_depth = Counter()
    by_write_loc = Counter()
    spans_instructions = 0
    len_wasted_bytes = Counter()
    noop_writes = 0
    real_writes = 0
    wasted_bytes = 0
    total = 0

    for line in args.log:
        if LINE not in line:
            continue
        event = parse_line(line)
        if args.owner and event.get("owner") != args.owner:
            continue
        if args.cause and event.get("cause") != args.cause:
            continue

        total += 1
        address = event.get("address", "?")
        data_len = event.get("data_len", -1)
        lamports = event.get("lamports", 0)

        causes[event.get("cause", "?")] += 1
        by_address[address] += 1
        by_data_len[data_len] += 1
        len_addresses[data_len].add(address)
        if data_len >= 0 and lamports == rent_exempt_minimum(data_len):
            len_at_rent_min[data_len] += 1
        per_slot[event.get("slot", -1)] += 1
        if "batch_len" in event:
            batch_lens.append(event["batch_len"])

        # changes_data_noop is a subset of changes_data, so real writes are the difference
        data_writes = event.get("changes_data", 0)
        noop = event.get("changes_data_noop", 0)
        writes_per_event[data_writes] += 1
        noop_writes += noop
        real_writes += max(0, data_writes - noop)
        if data_len >= 0:
            len_wasted_bytes[data_len] += noop * data_len
            wasted_bytes += noop * data_len

        write_program = event.get("write_program", "-")
        write_depth = event.get("write_stack_height", event.get("write_depth", "-"))
        by_write_program[write_program]["total"] += 1
        by_write_program[write_program][f"depth{write_depth}"] += 1
        by_write_depth[write_depth] += 1
        for chunk in event.get("write_locs", "-").split(","):
            loc, _, count = chunk.rpartition("x")
            if loc and count.isdigit():
                by_write_loc[loc] += int(count)
        # The account was written by more than one instruction. That is a round trip only when
        # the writes really changed the bytes; several instructions each writing it back
        # unchanged lands here too, and is the more common shape.
        if event.get("write_ix", "-") != event.get("write_ix_last", "-"):
            spans_instructions += 1

        address_meta.setdefault(address, (data_len, lamports))

    if not total:
        print("no matching events (check --owner/--cause, and that the log has "
              "'unmodified account written in lt_hash update' lines)", file=sys.stderr)
        return 1

    print(f"owner {args.owner or '(all owners)'}"
          + (f"   cause {args.cause}" if args.cause else ""))
    print(f"  {total} events over {len(per_slot)} slots, {len(by_address)} distinct accounts")
    print("  causes:         " + "  ".join(f"{c}={n}" for c, n in causes.most_common()))
    print(f"  events/slot:    {spread(list(per_slot.values()))}")
    print(f"  events/account: {spread(list(by_address.values()))}")
    if batch_lens:
        print(f"  batch_len:      {spread(batch_lens)}")
    print()

    print("data_len distribution   (one size is usually one account type)")
    print(f"  {'bytes':>8}  {'events':>12}  {'share':>6}  {'accounts':>9}  "
          f"{'rent_min':>14}  {'at rent_min':>11}  {'wasted':>10}")
    for data_len, n in by_data_len.most_common(args.top):
        rent_min = rent_exempt_minimum(data_len) if data_len >= 0 else 0
        print(f"  {data_len:>8}  {n:>12}  {pct(n, total)}  {len(len_addresses[data_len]):>9}  "
              f"{rent_min:>14}  {pct(len_at_rent_min[data_len], n)}  "
              f"{human_bytes(len_wasted_bytes[data_len]):>10}")
    print()

    if noop_writes or real_writes:
        print("write-back multiplicity   (data writes per event; >1 means CPI plus "
              "instruction return)")
        for count, n in sorted(writes_per_event.items()):
            print(f"  {count} write(s)  {n:>12}  {pct(n, total)}")
        print()

        print("wasted work")
        print(f"  no-op data writes      {noop_writes:>14}"
              f"   ({noop_writes / total:.2f} per event)")
        print(f"  real data writes       {real_writes:>14}")
        print(f"  bytes copied for free  {wasted_bytes:>14}   ({human_bytes(wasted_bytes)})")
        if len(per_slot):
            print(f"  per slot               {wasted_bytes / len(per_slot):>14.0f}   "
                  f"({human_bytes(wasted_bytes / len(per_slot))}/slot, "
                  f"{noop_writes / len(per_slot):.1f} writes/slot)")
        print("  each no-op write on a shared buffer is one allocation plus a memcpy of that size")
        print()

    if by_write_loc:
        print("write locations   (which code path performed the write)")
        print(f"  {'writes':>12}  {'share':>6}  location")
        loc_total = sum(by_write_loc.values())
        for loc, count in by_write_loc.most_common(args.top):
            print(f"  {count:>12}  {pct(count, loc_total)}  {loc}")
        print()

    if any(k != "-" for k in by_write_depth):
        print("instruction stack height   (1 = top-level, 2+ = inside CPI)")
        for depth, n in sorted(by_write_depth.items(), key=lambda kv: str(kv[0])):
            label = {"1": "1  top-level instruction", "-": "-  not captured"}.get(
                str(depth), f"{depth}  inside CPI")
            print(f"  {label:<32} {n:>12}  {pct(n, total)}")
        if spans_instructions:
            print(f"  {'writes from different instructions':<32} {spans_instructions:>12}  "
                  f"{pct(spans_instructions, total)}")
        print()

        print(f"top {args.top} writing programs   (executing program, not the account owner)")
        print(f"  {'events':>12}  {'share':>6}  program")
        ranked = sorted(by_write_program.items(), key=lambda kv: -kv[1]["total"])[: args.top]
        for program, counts in ranked:
            depths = " ".join(f"{k}={v}" for k, v in sorted(counts.items()) if k != "total")
            print(f"  {counts['total']:>12}  {pct(counts['total'], total)}  {program}  {depths}")
        print()

    print(f"top {args.top} accounts")
    print(f"  {'events':>12}  {'share':>6}  {'bytes':>6}  {'lamports':>16}  "
          f"{'over rent_min':>15}  address")
    for address, n in by_address.most_common(args.top):
        data_len, lamports = address_meta[address]
        over = lamports - rent_exempt_minimum(data_len) if data_len >= 0 else 0
        marker = "at minimum" if over == 0 else f"{over:+}"
        print(f"  {n:>12}  {pct(n, total)}  {data_len:>6}  {lamports:>16}  "
              f"{marker:>15}  {address}")
    print()

    ranked = [n for _, n in by_address.most_common()]
    print("concentration")
    for k in (1, 10, 100, 1000, 10000):
        if k <= len(ranked):
            print(f"  top {k:>6} accounts  {pct(sum(ranked[:k]), total)} of events")
    print(f"  all {len(ranked):>6} accounts  100.0% of events")

    if args.addresses:
        with open(args.addresses, "w") as handle:
            for address, _ in by_address.most_common():
                handle.write(address + "\n")
        print(f"\nwrote {len(by_address)} addresses to {args.addresses}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
