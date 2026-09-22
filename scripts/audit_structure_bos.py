from collections import Counter
from pathlib import Path

from market_engine.data import load_mt5_csv, validate_ohlcv
from market_engine.structure import (
    build_structural_sequence,
    process_structural_candles,
)

DATA = Path("data/raw/XAUUSDc_M30_202409012200_202609182030.csv")


def main():
    df = load_mt5_csv(DATA)
    validate_ohlcv(df)

    structural = build_structural_sequence(df)
    state, events = process_structural_candles(structural)

    bos = [
        e for e in events
        if e.event in {"BULLISH_BOS", "BEARISH_BOS"}
    ]

    print(f"raw candles       : {len(df):,}")
    print(f"structural candles: {len(structural):,}")
    print(f"structure events  : {len(events):,}")
    print(f"BOS events        : {len(bos):,}")
    print()

    counts = Counter(e.event for e in bos)
    print("BOS by direction:")
    for k, v in sorted(counts.items()):
        print(f"  {k:16} {v:,}")

    print()
    print("BOS by year:")
    yearly = Counter(e.timestamp.year for e in bos)
    for k, v in sorted(yearly.items()):
        print(f"  {k}: {v:,}")

    print()
    print("Last 20 BOS:")
    for e in bos[-20:]:
        print(
            f"  {e.timestamp} | "
            f"{e.event:13} | "
            f"index={e.index:5} | "
            f"swing_index={e.swing_index:5} | "
            f"price={e.swing_price}"
        )

    print()
    print("BOS candles with multiple events:")
    by_index = {}
    for e in bos:
        by_index.setdefault(e.index, []).append(e)

    multi = {k: v for k, v in by_index.items() if len(v) > 1}
    print(f"  candles with >1 BOS: {len(multi):,}")

    for idx, events_at_idx in list(multi.items())[:10]:
        print(f"  index={idx}")
        for e in events_at_idx:
            print(f"    {e.event:13} | swing_index={e.swing_index:5} | price={e.swing_price}")

    print()
    print("BOS spacing:")
    bos_indices = sorted(set(e.index for e in bos))
    if len(bos_indices) > 1:
        gaps = [b - a for a, b in zip(bos_indices, bos_indices[1:])]
        print(f"  min index gap   : {min(gaps)}")
        print(f"  median index gap: {sorted(gaps)[len(gaps)//2]}")
        print(f"  max index gap   : {max(gaps)}")

    print()
    print("Anti-lookahead audit:")
    violations = []
    for e in bos:
        if e.swing_index is None or e.swing_index >= e.index:
            violations.append((e.index, e.swing_index))

    print(f"  BOS events checked: {len(bos):,}")
    print(f"  violations        : {len(violations):,}")
    if violations:
        for item in violations[:10]:
            print(f"    {item}")


if __name__ == "__main__":
    main()
