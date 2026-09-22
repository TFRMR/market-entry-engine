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
    swings, events = process_structural_candles(structural)

    bos = [
        e for e in events
        if e.event in {"BULLISH_BOS", "BEARISH_BOS"}
    ]

    confirmations = {
        s.index: s.confirmation_index
        for s in swings
        if s.index is not None
    }

    violations = []

    for e in bos:
        confirmation_index = confirmations.get(e.swing_index)

        if confirmation_index is None or confirmation_index >= e.index:
            violations.append(
                (e.index, e.swing_index, confirmation_index)
            )

    print("Swing-confirmation audit:")
    print(f"  BOS checked : {len(bos):,}")
    print(f"  violations  : {len(violations):,}")

    if violations:
        print()
        print("First violations:")
        for item in violations[:10]:
            print(f"  BOS={item[0]} swing={item[1]} confirmation={item[2]}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
