from dataclasses import dataclass

@dataclass(frozen=True)
class TimeSeriesFold:
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int

def build_time_series_split(n_samples: int, validation_size: int, test_size: int, purge: int, embargo: int = 0) -> TimeSeriesFold:
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
    return TimeSeriesFold(0, train_end, validation_start, validation_end, test_start, n_samples)
