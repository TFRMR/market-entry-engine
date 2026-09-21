"""Realized-R analysis for structural setups.

Every setup gets a realized R after spread: stop and target exits use the same
deterministic semantics as evaluate_trade, and setups still open at the end of
the horizon exit at the close of the last horizon candle (time-stop).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_engine.entry import SetupCandidate
from market_engine.execution import execute_entry
from market_engine.exits import ExitAreaIndex, build_exit_areas
from market_engine.outcome import evaluate_trade
from market_engine.setup_facts import SetupFactContext
from market_engine.structure import Direction, StructureEvent, ValidSwing

REALIZED_COLUMNS = (
    "setup_index",
    "setup_timestamp",
    "entry_index",
    "direction",
    "exit_kind",
    "realized_r",
    "reward_r_after_spread",
    "risk_dist",
    "reward_dist",
    "ambiguous_barrier",
    "setup_bos_external",
)


def build_realized_r_frame(
    candidates: list[SetupCandidate],
    swings: list[ValidSwing],
    events: list[StructureEvent],
    frame: pd.DataFrame,
    spread_price: float,
    horizon: int,
) -> pd.DataFrame:
    """Return one row per evaluable setup with realized R after spread."""
    if horizon <= 0:
        raise ValueError("horizon must be greater than zero.")

    exit_index = ExitAreaIndex.build(frame)
    fact_context = SetupFactContext.build(swings, events)
    rows: list[dict[str, object]] = []

    for candidate in candidates:
        if candidate.setup_index + horizon >= len(frame):
            continue

        exit_areas = build_exit_areas(candidate, swings, frame, exit_index)
        if not exit_areas:
            continue

        target = exit_areas[0]
        execution = execute_entry(
            direction=candidate.direction,
            quoted_price=float(candidate.entry_price),
            spread_price=spread_price,
            invalidation_price=float(candidate.invalidation_price),
        )
        outcome = evaluate_trade(
            candidate=candidate,
            execution=execution,
            target=target,
            frame=frame,
            horizon=horizon,
        )

        is_up = candidate.direction is Direction.UP
        if outcome.status == "OPEN":
            close = float(frame["close"].iloc[candidate.setup_index + horizon])
            pnl = close - execution.entry_price if is_up else execution.entry_price - close
            realized_r = pnl / execution.risk
            exit_kind = "TIME_STOP"
        else:
            realized_r = float(outcome.risk_multiple)
            exit_kind = outcome.status

        signed_reward = (
            float(target.price) - execution.entry_price
            if is_up
            else execution.entry_price - float(target.price)
        )
        rows.append(
            {
                "setup_index": candidate.setup_index,
                "setup_timestamp": candidate.setup_timestamp,
                "entry_index": candidate.entry_index,
                "direction": candidate.direction.value,
                "exit_kind": exit_kind,
                "realized_r": float(realized_r),
                "reward_r_after_spread": signed_reward / execution.risk,
                "risk_dist": abs(
                    float(candidate.entry_price) - float(candidate.invalidation_price)
                ),
                "reward_dist": abs(float(target.price) - float(candidate.entry_price)),
                "ambiguous_barrier": bool(outcome.both_hit),
                "setup_bos_external": fact_context.facts(candidate)["setup_bos_external"],
            }
        )

    return pd.DataFrame(rows, columns=list(REALIZED_COLUMNS))


def simulate_fixed_geometry_r(
    frame: pd.DataFrame,
    entry_indices: np.ndarray,
    is_up: np.ndarray,
    risk_dist: np.ndarray,
    reward_dist: np.ndarray,
    spread_price: float,
    horizon: int,
) -> np.ndarray:
    """Realized R for trades with fixed stop/target distances at given entries.

    Entry is the open of each entry candle (plus/minus spread), stop and target
    sit at the given price distances from the quoted entry, stop wins a
    same-candle tie, and unresolved trades exit at the close of the last
    horizon candle. Semantics mirror build_realized_r_frame.
    """
    entry_indices = np.asarray(entry_indices, dtype=int)
    is_up = np.asarray(is_up, dtype=bool)
    risk_dist = np.asarray(risk_dist, dtype=float)
    reward_dist = np.asarray(reward_dist, dtype=float)

    if horizon <= 0:
        raise ValueError("horizon must be greater than zero.")
    if not (
        len(entry_indices)
        == len(is_up)
        == len(risk_dist)
        == len(reward_dist)
    ):
        raise ValueError("fixed-geometry arrays must have the same length.")
    if np.any(entry_indices < 0):
        raise ValueError("entry_indices must be non-negative.")
    if np.any(risk_dist <= 0) or np.any(reward_dist < 0):
        raise ValueError("risk_dist must be positive and reward_dist must be non-negative.")
    if spread_price < 0:
        raise ValueError("spread_price must not be negative.")

    opens = frame["open"].to_numpy(float)
    highs = frame["high"].to_numpy(float)
    lows = frame["low"].to_numpy(float)
    closes = frame["close"].to_numpy(float)

    window = entry_indices[:, None] + np.arange(horizon)[None, :]
    if window.size and window.max() >= len(frame):
        raise ValueError("entry window extends beyond the frame.")

    quoted = opens[entry_indices]
    sign = np.where(is_up, 1.0, -1.0)
    entry = quoted + sign * spread_price
    stop = quoted - sign * risk_dist
    target = quoted + sign * reward_dist

    high_w = highs[window]
    low_w = lows[window]
    stop_hit = np.where(
        is_up[:, None], low_w <= stop[:, None], high_w >= stop[:, None]
    )
    target_hit = np.where(
        is_up[:, None], high_w >= target[:, None], low_w <= target[:, None]
    )

    has_stop = stop_hit.any(axis=1)
    has_target = target_hit.any(axis=1)
    first_stop = np.where(has_stop, stop_hit.argmax(axis=1), horizon)
    first_target = np.where(has_target, target_hit.argmax(axis=1), horizon)

    resolved_stop = has_stop & (~has_target | (first_stop <= first_target))
    resolved_target = has_target & ~resolved_stop

    risk_exec = np.abs(entry - stop)
    pnl_stop = sign * (stop - entry)
    pnl_target = sign * (target - entry)
    pnl_time = sign * (closes[entry_indices + horizon - 1] - entry)

    pnl = np.where(
        resolved_stop,
        pnl_stop,
        np.where(resolved_target, pnl_target, pnl_time),
    )
    return pnl / risk_exec


def placebo_mean_r(
    frame: pd.DataFrame,
    is_up: np.ndarray,
    risk_dist: np.ndarray,
    reward_dist: np.ndarray,
    spread_price: float,
    horizon: int,
    entry_low: int,
    entry_high: int,
    draws: int,
    seed: int,
) -> np.ndarray:
    """Mean R of random-timing placebos with each real setup's geometry.

    Each draw gives every real setup a uniformly random entry candle in
    [entry_low, entry_high], keeping its direction and stop/target distances.
    """
    if entry_high < entry_low:
        raise ValueError("entry_high must not be below entry_low.")
    if draws <= 0:
        raise ValueError("draws must be greater than zero.")
    if len(is_up) != len(risk_dist) or len(is_up) != len(reward_dist):
        raise ValueError("placebo geometry arrays must have the same length.")

    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    count = len(is_up)
    for draw in range(draws):
        entries = rng.integers(entry_low, entry_high + 1, size=count)
        means[draw] = simulate_fixed_geometry_r(
            frame,
            entries,
            is_up,
            risk_dist,
            reward_dist,
            spread_price,
            horizon,
        ).mean()
    return means


def cluster_bootstrap_mean(
    values: np.ndarray,
    clusters: np.ndarray,
    draws: int = 2000,
    seed: int = 7,
) -> tuple[float, float, float]:
    """Mean with a 95% cluster-bootstrap interval (resampling whole clusters)."""
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    if values.size != clusters.size:
        raise ValueError("values and clusters must have the same length.")
    if draws <= 0:
        raise ValueError("draws must be greater than zero.")

    codes, _ = pd.factorize(pd.Series(clusters))
    sums = np.bincount(codes, weights=values)
    counts = np.bincount(codes).astype(float)

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(sums), size=(draws, len(sums)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )
