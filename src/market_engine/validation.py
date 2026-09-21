"""Deterministic leakage-safe time-series validation splits."""

from __future__ import annotations

from dataclasses import dataclass

from market_engine.labels import STRUCTURAL_LABEL_HORIZON


@dataclass(frozen=True)
class TimeSeriesFold:
    """One chronological train/validation/test split represented by indices."""

    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int

    @property
    def train_indices(self) -> range:
        return range(self.train_start, self.train_end)

    @property
    def validation_indices(self) -> range:
        return range(self.validation_start, self.validation_end)

    @property
    def test_indices(self) -> range:
        return range(self.test_start, self.test_end)


def build_time_series_split(
    n_samples: int,
    validation_size: int,
    test_size: int,
    purge: int = STRUCTURAL_LABEL_HORIZON,
    embargo: int = 0,
) -> TimeSeriesFold:
    """Build one expanding-train chronological split with purge and embargo.

    Purge removes training samples whose forward label windows could overlap
    validation. Embargo adds an explicit empty gap between validation and test.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be greater than zero.")
    if validation_size <= 0 or test_size <= 0:
        raise ValueError("validation_size and test_size must be greater than zero.")
    if purge < 0 or embargo < 0:
        raise ValueError("purge and embargo must not be negative.")

    test_start = n_samples - test_size
    validation_end = test_start - embargo
    validation_start = validation_end - validation_size
    train_end = validation_start - purge

    if train_end <= 0 or validation_start < 0 or test_start < 0:
        raise ValueError("Not enough samples for the requested purge and split sizes.")

    return TimeSeriesFold(
        train_start=0,
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end,
        test_start=test_start,
        test_end=n_samples,
    )
