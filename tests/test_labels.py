import pandas as pd
import pytest

from market_engine.entry import SetupCandidate
from market_engine.labels import (
    ALL_BARRIER_LABELS,
    BOTH_SAME_CANDLE,
    MODEL_LABELS,
    SL_FIRST,
    STRUCTURAL_LABEL_HORIZON,
    TP_FIRST,
    UNRESOLVED,
    build_setup_label_dataset,
    is_model_label,
)
from market_engine.outcomes import add_barrier_outcomes
from market_engine.structure import Direction, SwingType, ValidSwing


def make_candidate() -> SetupCandidate:
    return SetupCandidate(
        setup_index=1,
        setup_timestamp=1,
        direction=Direction.UP,
        entry_index=2,
        entry_timestamp=2,
        entry_price=100.0,
        invalidation_price=95.0,
        risk=5.0,
        invalidation_swing_index=0,
    )


def make_swings() -> list[ValidSwing]:
    return [
        ValidSwing(
            swing_type=SwingType.HIGH,
            index=0,
            timestamp=0,
            confirmation_index=0,
            confirmation_timestamp=0,
            price=110.0,
        )
    ]


def make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"timestamp": 0, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
            {"timestamp": 1, "open": 100.0, "high": 102.0, "low": 98.0, "close": 101.0},
            {"timestamp": 2, "open": 101.0, "high": 103.0, "low": 100.0, "close": 102.0},
            {"timestamp": 3, "open": 102.0, "high": 110.0, "low": 101.0, "close": 109.0},
            {"timestamp": 4, "open": 109.0, "high": 111.0, "low": 108.0, "close": 110.0},
        ]
    )


def test_barrier_label_contract_is_explicit() -> None:
    assert MODEL_LABELS == {TP_FIRST, SL_FIRST}
    assert ALL_BARRIER_LABELS == {
        TP_FIRST,
        SL_FIRST,
        BOTH_SAME_CANDLE,
        UNRESOLVED,
    }
    assert STRUCTURAL_LABEL_HORIZON == 10
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


def test_setup_label_uses_structural_target_and_setup_features() -> None:
    frame = make_frame()
    features = frame.copy()
    features["setup_feature"] = [10, 20, 30, 40, 50]

    result = build_setup_label_dataset(
        candidates=[make_candidate()],
        swings=make_swings(),
        frame=frame,
        feature_frame=features,
        spread_price=0.0,
        feature_columns=("setup_feature",),
        horizon=2,
    )

    assert len(result) == 1
    assert result.loc[0, "label"] == TP_FIRST
    assert result.loc[0, "target_price"] == 110.0
    assert result.loc[0, "setup_feature"] == 20
    assert result.loc[0, "entry_index"] == 2
    assert result.loc[0, "reward_risk"] == pytest.approx(2.0)
    assert result.loc[0, "ambiguous_barrier"] is False


def test_setup_label_maps_horizon_expiry_to_unresolved() -> None:
    frame = make_frame()
    frame.loc[3, "high"] = 105.0
    frame.loc[4, "high"] = 106.0

    features = frame.copy()
    features["setup_feature"] = [10, 20, 30, 40, 50]

    result = build_setup_label_dataset(
        candidates=[make_candidate()],
        swings=make_swings(),
        frame=frame,
        feature_frame=features,
        spread_price=0.0,
        feature_columns=("setup_feature",),
        horizon=2,
    )

    assert result.loc[0, "label"] == UNRESOLVED


def test_setup_label_maps_stop_to_sl_first() -> None:
    frame = make_frame()
    frame.loc[2, "low"] = 94.0

    features = frame.copy()
    features["setup_feature"] = [10, 20, 30, 40, 50]

    result = build_setup_label_dataset(
        candidates=[make_candidate()],
        swings=make_swings(),
        frame=frame,
        feature_frame=features,
        spread_price=0.0,
        feature_columns=("setup_feature",),
        horizon=2,
    )

    assert result.loc[0, "label"] == SL_FIRST


def test_setup_label_requires_positive_horizon() -> None:
    frame = make_frame()
    features = frame.copy()
    features["setup_feature"] = [10, 20, 30, 40, 50]

    with pytest.raises(ValueError, match="horizon"):
        build_setup_label_dataset(
            candidates=[make_candidate()],
            swings=make_swings(),
            frame=frame,
            feature_frame=features,
            spread_price=0.0,
            feature_columns=("setup_feature",),
            horizon=0,
        )


def test_setup_label_skips_incomplete_forward_horizon() -> None:
    frame = make_frame().iloc[:3].copy()
    features = frame.copy()
    features["setup_feature"] = [10, 20, 30]

    result = build_setup_label_dataset(
        candidates=[make_candidate()],
        swings=make_swings(),
        frame=frame,
        feature_frame=features,
        spread_price=0.0,
        feature_columns=("setup_feature",),
        horizon=2,
    )

    assert result.empty
