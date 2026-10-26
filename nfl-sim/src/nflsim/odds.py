"""Odds conversion, vig removal, fair odds and expected profit.

Conventions: American odds are ints/floats like -110 or +150. Decimal odds
include the stake (2.0 = even money). Probabilities are in [0, 1].
"""
from __future__ import annotations

import numpy as np


def american_to_decimal(a):
    a = np.asarray(a, dtype=float)
    if np.any(np.abs(a) < 100):
        raise ValueError("American odds must be <= -100 or >= +100")
    return np.where(a > 0, 1 + a / 100.0, 1 + 100.0 / -a)


def decimal_to_american(d):
    d = np.asarray(d, dtype=float)
    if np.any(d <= 1):
        raise ValueError("decimal odds must be > 1")
    return np.where(d >= 2, (d - 1) * 100.0, -100.0 / (d - 1))


def decimal_to_prob(d):
    return 1.0 / np.asarray(d, dtype=float)


def american_to_prob(a):
    return decimal_to_prob(american_to_decimal(a))


def prob_to_decimal(p):
    p = np.asarray(p, dtype=float)
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("probability must be in (0, 1)")
    return 1.0 / p


def prob_to_american(p):
    return decimal_to_american(prob_to_decimal(p))


def devig(odds_a, odds_b, method: str = "multiplicative", fmt: str = "american"):
    """No-vig probabilities for a two-sided market.

    multiplicative: divide each implied probability by their sum.
    power: find k with p_a**k + p_b**k = 1 (shades longshots more).
    Returns (p_a, p_b, overround) where overround = sum of implied probs - 1.
    """
    conv = american_to_prob if fmt == "american" else decimal_to_prob
    pa, pb = float(conv(odds_a)), float(conv(odds_b))
    over = pa + pb - 1
    if method == "multiplicative":
        return pa / (pa + pb), pb / (pa + pb), over
    if method == "power":
        lo, hi = 1e-6, 50.0
        for _ in range(200):
            k = (lo + hi) / 2
            s = pa ** k + pb ** k
            lo, hi = (lo, k) if s < 1 else (k, hi)
        return pa ** k, pb ** k, over
    raise ValueError(method)


def fair_odds(p: float) -> dict:
    """Odds with no vig for an event of probability p (push-free)."""
    return {"prob": float(p), "decimal": float(prob_to_decimal(p)),
            "american": float(prob_to_american(p))}


def expected_value(p_win: float, decimal_odds: float, p_push: float = 0.0) -> dict:
    """Expected profit per 1 unit staked. A push returns the stake.

    profit = p_win * (decimal - 1) - p_loss, with p_loss = 1 - p_win - p_push.
    ROI is profit per unit staked (the same number, as a fraction).
    """
    if not 0 <= p_win <= 1 or not 0 <= p_push <= 1 - p_win + 1e-12:
        raise ValueError("invalid probabilities")
    p_loss = max(0.0, 1.0 - p_win - p_push)
    ev = p_win * (decimal_odds - 1.0) - p_loss
    breakeven = 1.0 / decimal_odds * (1 - p_push)  # win prob needed for EV = 0
    return {"ev_per_unit": float(ev), "roi": float(ev), "p_loss": float(p_loss),
            "breakeven_p_win": float(breakeven)}


def expected_value_payouts(payouts: np.ndarray) -> dict:
    """EV from per-simulation gross returns per unit stake (0 = loss, 1 = refund)."""
    payouts = np.asarray(payouts, dtype=float)
    ev = payouts.mean() - 1.0
    se = payouts.std(ddof=1) / np.sqrt(len(payouts)) if len(payouts) > 1 else np.nan
    return {"ev_per_unit": float(ev), "roi": float(ev), "mc_se": float(se)}
