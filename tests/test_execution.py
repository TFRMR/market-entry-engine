import pytest

from market_engine.execution import execute_entry
from market_engine.structure import Direction


def test_long_entry_pays_spread_and_risk_uses_executed_price():
    execution = execute_entry(
        Direction.UP,
        quoted_price=100.0,
        spread_price=0.2,
        invalidation_price=95.0,
    )

    assert execution.entry_price == 100.2
    assert execution.risk == pytest.approx(5.2)


def test_short_entry_pays_spread_and_risk_uses_executed_price():
    execution = execute_entry(
        Direction.DOWN,
        quoted_price=100.0,
        spread_price=0.2,
        invalidation_price=105.0,
    )

    assert execution.entry_price == 99.8
    assert execution.risk == pytest.approx(5.2)


def test_zero_spread_preserves_quoted_price():
    execution = execute_entry(
        Direction.UP,
        quoted_price=100.0,
        spread_price=0.0,
        invalidation_price=95.0,
    )

    assert execution.entry_price == 100.0
    assert execution.risk == pytest.approx(5.0)


def test_negative_spread_is_rejected():
    with pytest.raises(ValueError, match="Spread"):
        execute_entry(
            Direction.UP,
            quoted_price=100.0,
            spread_price=-0.1,
            invalidation_price=95.0,
        )


def test_non_positive_execution_risk_is_rejected():
    with pytest.raises(ValueError, match="risk"):
        execute_entry(
            Direction.UP,
            quoted_price=100.0,
            spread_price=0.0,
            invalidation_price=100.0,
        )
