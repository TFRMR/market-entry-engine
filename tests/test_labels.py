import pandas as pd

from market_engine.labels import (
    ALL_BARRIER_LABELS,
    BOTH_SAME_CANDLE,
    MODEL_LABELS,
    SL_FIRST,
    TP_FIRST,
    UNRESOLVED,
    is_model_label,
)
from market_engine.outcomes import add_barrier_outcomes


def test_barrier_label_contract_is_explicit() -> None:
    assert MODEL_LABELS == {TP_FIRST, SL_FIRST}
    assert ALL_BARRIER_LABELS == {
        TP_FIRST,
        SL_FIRST,
        BOTH_SAME_CANDLE,
        UNRESOLVED,
    }
    assert is_model_label(TP_FIRST)
    assert is_model_label(SL_FIRST)
    assert not is_model_label(BOTH_SAME_CANDLE)
    assert not is_model_label(UNRESOLVED)


def test_barrier_generation_uses_canonical_labels() -> None:
    frame = pd.DataFrame(
        {
            "open": [100.0, 104.0],
            "high": [105.0, 110.0],
            "low": [99.0, 103.0],
            "close": [104.0, 109.0],
            "is_momentum_candle": [True, False],
        }
    )

    result = add_barrier_outcomes(
        frame,
        horizons=(1,),
        targets_r=(1.0,),
    )

    assert result.loc[0, "barrier_1r_1"] == TP_FIRST
