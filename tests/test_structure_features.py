import pandas as pd

from market_engine.features import add_structure_event_features


def make_structure_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=7, freq="30min"),
            "open": [10, 11, 13, 13, 11, 10, 12],
            "high": [12, 14, 14.5, 13.5, 12, 13, 15],
            "low": [9, 10, 11, 10, 9, 10, 8],
            "close": [11, 13, 14, 11, 10, 12, 14],
        }
    )


def test_structure_features_are_available_only_from_confirmation_time():
    result = add_structure_event_features(make_structure_frame())

    assert result.loc[3, "structure_swing_high_valid"] == 0
    assert result.loc[4, "structure_swing_high_valid"] == 1
    assert pd.isna(result.loc[3, "structure_last_valid_high"])
    assert result.loc[4, "structure_last_valid_high"] == 14.5
    assert result.loc[5, "structure_last_valid_high"] == 14.5


def test_structure_features_expose_bos_and_swing_flags():
    result = add_structure_event_features(make_structure_frame())

    assert result.loc[6, "structure_bullish_bos"] == 1
    assert result.loc[6, "structure_bearish_bos"] == 0
    assert result.loc[4, "structure_internal_swing"] == 0
    assert result.loc[4, "structure_external_swing"] == 1


def test_event_sequence_features_are_point_in_time():
    from market_engine.features import add_event_sequence_features

    result = add_event_sequence_features(make_structure_frame())

    # The first confirmed swing appears at index 4. Earlier rows must
    # not know about it.
    assert result.loc[3, "structure_event_count"] == 0
    assert pd.isna(result.loc[3, "structure_last_event_index"])
    assert pd.isna(result.loc[3, "structure_last_event_type"])

    assert result.loc[4, "structure_event_count"] == 1
    assert result.loc[4, "structure_last_event_index"] == 4
    assert result.loc[4, "structure_last_event_type"] == "SWING_HIGH_VALID"
    assert result.loc[4, "structure_last_event_age"] == 0

    # The event is carried forward after confirmation.
    assert result.loc[5, "structure_event_count"] == 1
    assert result.loc[5, "structure_last_event_index"] == 4
    assert result.loc[5, "structure_last_event_age"] == 1


def test_event_sequence_preserves_feature_layer_event_order():
    from market_engine.features import add_event_sequence_features

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=8, freq="30min"),
            "open": [10, 11, 13, 13, 11, 9, 9, 10],
            "high": [12, 14, 14.5, 13.5, 13, 10, 11, 16],
            "low": [9, 10, 11, 10, 8, 9, 8.5, 7],
            "close": [11, 13, 14, 11, 9, 9.5, 10.5, 14],
        }
    )

    result = add_event_sequence_features(frame)

    assert result.loc[7, "structure_last_event_type"] == "BULLISH_CHOCH"
    assert result.loc[7, "structure_previous_event_type"] == "BULLISH_BOS"
    assert result.loc[7, "structure_event_count"] == 3
    assert result.loc[7, "structure_events_since_last_bos"] == 0



def test_event_sequence_tracks_bos_and_swing_distances():
    from market_engine.features import add_event_sequence_features

    result = add_event_sequence_features(make_structure_frame())

    assert result.loc[6, "structure_bars_since_last_bos"] == 0
    assert result.loc[6, "structure_events_since_last_bos"] == 0

    assert result.loc[6, "structure_bars_since_last_swing"] == 2
    assert result.loc[6, "structure_events_since_last_swing"] >= 0
