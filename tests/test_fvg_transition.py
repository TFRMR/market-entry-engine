from __future__ import annotations

import pandas as pd
import pytest

from market_engine.fvg_transition import (
    FVGReference,
    build_fvg_references,
)


def test_fvg_reference_is_immutable() -> None:
    reference = FVGReference(
        creation_index=2,
        direction="BULLISH",
        lower=10.0,
        upper=11.0,
        size=1.0,
    )

    with pytest.raises((AttributeError, TypeError)):
        reference.lower = 12.0  # type: ignore[misc]


def test_build_fvg_references_preserves_all_historical_fvgs() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=7,
                freq="30min",
            ),
            "high": [
                10.0,
                11.0,
                12.0,
                14.0,
                15.0,
                13.0,
                11.0,
            ],
            "low": [
                8.0,
                9.0,
                11.0,
                13.0,
                14.0,
                10.0,
                7.0,
            ],
        }
    )

    references = build_fvg_references(frame)

    assert len(references) == 4

    assert references[0].creation_index == 2
    assert references[0].direction == "BULLISH"
    assert references[0].lower == 10.0
    assert references[0].upper == 11.0
    assert references[0].size == 1.0

    assert references[1].creation_index == 3
    assert references[1].direction == "BULLISH"
    assert references[1].lower == 11.0
    assert references[1].upper == 13.0
    assert references[1].size == 2.0

    assert references[2].creation_index == 4
    assert references[2].direction == "BULLISH"
    assert references[2].lower == 12.0
    assert references[2].upper == 14.0
    assert references[2].size == 2.0

    assert references[3].creation_index == 6
    assert references[3].direction == "BEARISH"
    assert references[3].lower == 11.0
    assert references[3].upper == 14.0
    assert references[3].size == 3.0


def test_fvg_reference_becomes_available_on_creation_candle() -> None:
    frame = pd.DataFrame(
        {
            "high": [10.0, 11.0, 12.0],
            "low": [8.0, 9.0, 11.0],
        }
    )

    references = build_fvg_references(frame)

    assert len(references) == 1
    assert references[0].creation_index == 2


def test_fvg_reference_does_not_use_future_candles() -> None:
    frame = pd.DataFrame(
        {
            "high": [10.0, 11.0, 12.0, 14.0],
            "low": [8.0, 9.0, 11.0, 13.0],
        }
    )

    references = build_fvg_references(frame)

    assert [reference.creation_index for reference in references] == [2, 3]


def test_fvg_reference_requires_ohlc_context() -> None:
    frame = pd.DataFrame({"close": [100.0]})

    with pytest.raises(
        ValueError,
        match="Missing FVG reference columns",
    ):
        build_fvg_references(frame)
