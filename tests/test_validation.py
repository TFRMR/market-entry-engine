import pytest

from market_engine.validation import build_time_series_split


def test_time_series_split_is_chronological_and_purged() -> None:
    fold = build_time_series_split(100, validation_size=20, test_size=20, embargo=3)
    assert list(range(fold.train_start, fold.train_end)) == list(range(47))
    assert list(range(fold.validation_start, fold.validation_end)) == list(range(57, 77))
    assert list(range(fold.test_start, fold.test_end)) == list(range(80, 100))
    assert fold.train_end + 10 == fold.validation_start
    assert fold.validation_end + 3 == fold.test_start


def test_time_series_split_rejects_insufficient_history() -> None:
    with pytest.raises(ValueError, match="Not enough samples"):
        build_time_series_split(30, validation_size=10, test_size=10, purge=10)


def test_time_series_split_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        build_time_series_split(0, 5, 5, 1)
    with pytest.raises(ValueError, match="greater than zero"):
        build_time_series_split(30, 0, 5, 1)
    with pytest.raises(ValueError, match="negative"):
        build_time_series_split(30, 5, 5, -1)
