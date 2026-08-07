#!/usr/bin/env python3
"""Summarise unmodified-account events from a validator log.

Parses the lines emitted by `debug_log_unmodified_account()` when a validator runs with
AGAVE_DEBUG_UNMODIFIED_ACCOUNTS=1, and reports per-slot and aggregate statistics for
`bank-accounts_lt_hash.mean_num_accounts_unmodified`.

The question this answers: how much of the metric is *irreducible* (a field changed and changed
back within a transaction, which only a final-vs-initial comparison can catch) versus
*suppressible* (a byte-identical write that a comparison at the write site would avoid).

Usage:
    scripts/unmodified-accounts-stats.py validator.log
    journalctl -u agave -o cat | scripts/unmodified-accounts-stats.py
    scripts/unmodified-accounts-stats.py validator.log --by-owner --csv slots.csv
"""

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict

LINE = "unmodified account written in lt_hash update:"
KV = re.compile(r"(\w+)=(\S+)")

# Causes that no write-site comparison can eliminate: the account was genuinely changed and then
# changed back, so every individual write was a real change when it happened.
IRREDUCIBLE = ("data-roundtrip", "lamports-roundtrip")
# Suppressible at the write site by comparing before writing.
SUPPRESSIBLE = ("noop-data-write",)
# Neither: the account was written by several transactions whose net effect cancels, which is
# inherent to batching rather than to touch tracking.
BATCHING = ("batch-netting",)
# Attribution gaps worth investigating rather than reporting as a cause.
SUSPECT = ("unattributed", "inconsistent")

CAUSES = IRREDUCIBLE + SUPPRESSIBLE + BATCHING + SUSPECT

INT_FIELDS = (
    "slot",
    "lamports",
    "data_len",
    "num_occurrences",
    "batch_len",
    "changes_lamports",
    "changes_data",
    "changes_data_noop",
    "changes_data_len",
    "changes_owner",
)


def parse(stream):
    """Yields one dict per unmodified-account line; ignores everything else in the log."""
    for lineno, line in enumerate(stream, 1):
        if LINE not in line:
            continue
        event = dict(KV.findall(line[line.index(LINE) + len(LINE):]))
        if "slot" not in event or "cause" not in event:
            print(f"warning: line {lineno}: missing slot/cause, skipping", file=sys.stderr)
            continue
        for field in INT_FIELDS:
            if field in event:
                try:
                    event[field] = int(event[field])
                except ValueError:
                    print(
                        f"warning: line {lineno}: {field}={event[field]!r} is not an integer",
                        file=sys.stderr,
                    )
                    event[field] = 0
        yield event


def reclassify(event):
    """Recomputes the cause from the raw change counts, mirroring `debug_unmodified::classify`.

    Used to flag disagreement with the cause the validator logged, which would mean the two have
    drifted apart.
    """
    if event.get("num_occurrences", 1) > 1:
        return "batch-netting"
    real_data = max(0, event.get("changes_data", 0) - event.get("changes_data_noop", 0))
    if real_data >= 2 or event.get("changes_data_len", 0) >= 2:
        return "data-roundtrip"
    if event.get("changes_lamports", 0) >= 2:
        return "lamports-roundtrip"
    if event.get("changes_data_noop", 0) >= 1:
        return "noop-data-write"
    if real_data == 1 or event.get("changes_lamports", 0) == 1 or event.get("changes_data_len", 0) == 1:
        return "inconsistent"
    return "unattributed"


def pct(n, total):
    return f"{100.0 * n / total:5.1f}%" if total else "    -"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?", type=argparse.FileType("r", errors="replace"),
                    default=sys.stdin, help="validator log (default: stdin)")
    ap.add_argument("--by-owner", action="store_true", help="break down by owner program")
    ap.add_argument("--by-address", action="store_true", help="list the most frequent accounts")
    ap.add_argument("--top", type=int, default=15, help="rows in the owner/address tables")
    ap.add_argument("--slots", type=int, default=20, help="per-slot rows to print (0 for all)")
    ap.add_argument("--csv", help="write the full per-slot table to this file")
    args = ap.parse_args()

    per_slot = defaultdict(Counter)
    by_owner = defaultdict(Counter)
    by_address = Counter()
    unknown_causes = set()
    totals = Counter()
    addresses_per_slot = defaultdict(set)
    mismatches = 0

    for event in parse(args.log):
        slot, cause = event["slot"], event["cause"]
        if cause not in CAUSES:
            unknown_causes.add(cause)
        # Only cross-check when the raw counts are present; older logs predate them.
        if "changes_data_noop" in event and reclassify(event) != cause:
            mismatches += 1
        per_slot[slot][cause] += 1
        per_slot[slot]["total"] += 1
        totals[cause] += 1
        totals["total"] += 1
        addresses_per_slot[slot].add(event.get("address", "?"))
        by_owner[event.get("owner", "?")][cause] += 1
        by_owner[event.get("owner", "?")]["total"] += 1
        by_address[event.get("address", "?")] += 1

    if not totals["total"]:
        print("no unmodified-account events found "
              "(was the validator run with AGAVE_DEBUG_UNMODIFIED_ACCOUNTS=1?)", file=sys.stderr)
        return 1

    n = totals["total"]
    slots = sorted(per_slot)
    print(f"{n} events across {len(slots)} slots "
          f"(slots {slots[0]}..{slots[-1]}), {len(by_address)} distinct accounts")
    if mismatches:
        print(f"warning: {mismatches} events whose logged cause disagrees with the raw change "
              f"counts; classify() and this script may have drifted apart", file=sys.stderr)
    print()

    print("cause breakdown")
    for cause in CAUSES + tuple(sorted(unknown_causes)):
        if totals[cause]:
            note = "  (unknown to this script)" if cause in unknown_causes else ""
            print(f"  {cause:20s} {totals[cause]:8d}  {pct(totals[cause], n)}{note}")
    print()

    irreducible = sum(totals[c] for c in IRREDUCIBLE)
    suppressible = sum(totals[c] for c in SUPPRESSIBLE)
    batching = sum(totals[c] for c in BATCHING)
    suspect = sum(totals[c] for c in SUSPECT)
    unknown = sum(totals[c] for c in unknown_causes)
    print("what a fix could achieve")
    print(f"  irreducible (round trips)   {irreducible:8d}  {pct(irreducible, n)}"
          "   needs a final-vs-initial comparison")
    print(f"  suppressible (no-op writes) {suppressible:8d}  {pct(suppressible, n)}"
          "   fixable at the write site")
    if batching:
        print(f"  batch netting               {batching:8d}  {pct(batching, n)}"
              "   inherent to batching")
    if suspect:
        print(f"  unattributed/inconsistent   {suspect:8d}  {pct(suspect, n)}"
              "   attribution gap, investigate")
    if unknown:
        print(f"  unknown cause               {unknown:8d}  {pct(unknown, n)}"
              "   log predates this script, re-run to re-attribute")
    print()

    per_slot_counts = [per_slot[s]["total"] for s in slots]
    per_slot_counts.sort()
    mean = sum(per_slot_counts) / len(per_slot_counts)
    median = per_slot_counts[len(per_slot_counts) // 2]
    p95 = per_slot_counts[min(len(per_slot_counts) - 1, int(0.95 * len(per_slot_counts)))]
    print(f"per slot: mean {mean:.1f}  median {median}  p95 {p95}  "
          f"min {per_slot_counts[0]}  max {per_slot_counts[-1]}")
    print()

    shown = slots if args.slots == 0 else slots[: args.slots]
    all_causes = CAUSES + tuple(sorted(unknown_causes))
    header = ["slot", "total", "uniq"] + [c for c in all_causes if totals[c]]
    widths = [12, 6, 6] + [max(len(c), 6) for c in header[3:]]
    print("per-slot detail" + (f" (first {len(shown)} of {len(slots)})" if len(shown) < len(slots) else ""))
    print("  " + "  ".join(h.rjust(w) for h, w in zip(header, widths)))
    for slot in shown:
        row = [str(slot), str(per_slot[slot]["total"]), str(len(addresses_per_slot[slot]))]
        row += [str(per_slot[slot][c]) for c in header[3:]]
        print("  " + "  ".join(v.rjust(w) for v, w in zip(row, widths)))
    print()

    if args.by_owner:
        print(f"top {args.top} owner programs")
        ranked = sorted(by_owner.items(), key=lambda kv: -kv[1]["total"])[: args.top]
        for owner, counts in ranked:
            causes = " ".join(f"{c}={counts[c]}" for c in all_causes if counts[c])
            print(f"  {counts['total']:8d}  {pct(counts['total'], n)}  {owner}  {causes}")
        print()

    if args.by_address:
        print(f"top {args.top} accounts")
        for address, count in by_address.most_common(args.top):
            print(f"  {count:8d}  {pct(count, n)}  {address}")
        print()

    if args.csv:
        with open(args.csv, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["slot", "total", "unique_accounts"] + list(all_causes))
            for slot in slots:
                writer.writerow(
                    [slot, per_slot[slot]["total"], len(addresses_per_slot[slot])]
                    + [per_slot[slot][c] for c in all_causes]
                )
        print(f"wrote per-slot table for {len(slots)} slots to {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
