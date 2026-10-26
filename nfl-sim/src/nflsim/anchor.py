"""Market-anchored simulation mode.

The simulator's joint structure (how legs move together) is good, but its game
view is worse than the closing market. In market_anchored mode we keep every
part of the simulator and only shift each team's offensive strength so the
simulated MEAN home margin and MEAN total equal the market's spread and total.

What is shifted: an additive log-odds term on each offense's scoring outcomes
(touchdown and field goal) in the drive-outcome model, one per team. It
multiplies that offense's odds of a scoring drive by exp(shift). Nothing else
changes: player shares, play templates, game-state effects, pace, field
position and PAT behavior are untouched, so correlations still come from the
shared simulation.

(A first version shifted the `off_epa` input instead. In 69 of 1,530 backtest
games it failed to hit the targets: the needed shifts were up to 10 standard
deviations outside the training range and the logistic response saturated.
The log-odds shift is monotone in expected points and does not extrapolate
team ratings.)

Solving: two unknowns (home and away offense shift), two targets. Damped
Newton with a finite-difference Jacobian and Broyden updates, using common
random numbers (the same seed and N as the final run). The final run IS the
last solver evaluation, so the reported simulation hits the targets within
tolerance by construction, not by a separate noisy rerun.

Sign convention (verified on 2,877 games, see README): nflverse
`spread_line` is the expected HOME margin: positive = home favored. It agrees
in sign with the moneyline favorite in 98.1% of games (all disagreements are
|spread| <= 1.5 with near-even moneylines) and the slope of actual home margin
on spread_line is 1.03. A sportsbook spread for the home team is the negative
of this (home -3.5 <=> spread_line +3.5).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .ingest import MANIFEST
from .simulator import Matchup, simulate
from .team_model import OUT_IDX

MODES = ("independent", "market_anchored")
TOL = 0.25


@dataclass
class Line:
    target_margin: float | None      # expected home margin (nflverse convention)
    target_total: float | None
    source: str
    as_of: str | None
    note: str = ""
    warning: str = ""

    @property
    def complete(self) -> bool:
        return self.target_margin is not None and self.target_total is not None


def _schedule_downloaded_at(season: int) -> str | None:
    try:
        return json.loads(MANIFEST.read_text())["schedules"][str(season)]["downloaded_at"]
    except Exception:
        return None


def get_line(game_row: pd.Series, home_spread: float | None = None,
             total: float | None = None, snapshot: dict | None = None) -> Line:
    """Market line for a game. Priority, per component (spread and total):
    user value -> latest pre-kickoff odds snapshot (odds_feed.latest_consensus,
    median over books) -> nflverse schedules -> missing (caller falls back to
    independent mode).

    home_spread is the sportsbook spread for the HOME team (negative = home
    favored), as a bettor would type it. It is converted to the expected home
    margin = -home_spread. Never invents a line: missing values stay None.
    """
    src, notes, warn = [], [], ""
    snap_ok = snapshot is not None
    if snap_ok:
        snap_desc = (f"The Odds API {snapshot['source']} snapshot at "
                     f"{pd.Timestamp(snapshot['snapshot_time']):%Y-%m-%d %H:%M UTC}, "
                     f"median of {snapshot['n_books']} books")
        if snapshot.get("stale"):
            warn = (f"WARNING: odds snapshot is {snapshot['age_hours']:.1f} h old "
                    f"(> 6 h before {'kickoff' if snapshot.get('age_ref') == 'kickoff' else 'now/kickoff'})")
    if home_spread is not None:
        m = -float(home_spread)
        src.append(f"spread from user (home {home_spread:+g})")
    elif snap_ok and not pd.isna(snapshot.get("home_spread")):
        m = -float(snapshot["home_spread"])
        src.append(f"spread from {snap_desc} (home {snapshot['home_spread']:+g})")
    else:
        v = game_row.get("mkt_spread_line")
        m = None if v is None or pd.isna(v) else float(v)
        if m is not None:
            src.append("spread from nflverse schedules.spread_line")
    if total is not None:
        t = float(total)
        src.append("total from user")
    elif snap_ok and not pd.isna(snapshot.get("total")):
        t = float(snapshot["total"])
        src.append(f"total from {snap_desc}")
    else:
        v = game_row.get("mkt_total_line")
        t = None if v is None or pd.isna(v) else float(v)
        if t is not None:
            src.append("total from nflverse schedules.total_line")
    as_of = None
    if any("nflverse" in s for s in src):
        dl = _schedule_downloaded_at(int(game_row["season"]))
        played = not pd.isna(game_row.get("label_home_score", np.nan))
        kind = "closing line" if played else "latest line published by nflverse (game not played yet)"
        as_of = (f"{kind}; nflverse gives no line timestamp; schedule file downloaded {dl}"
                 if dl else f"{kind}; download time unknown")
    if m is None:
        notes.append("no spread available")
    if t is None:
        notes.append("no total available")
    if snap_ok and any("Odds API" in s for s in src):
        as_of = (as_of + "; " if as_of else "") + (f"odds snapshot pulled {pd.Timestamp(snapshot['pulled_at']):%Y-%m-%d %H:%M UTC}")
    if not any("Odds API" in s for s in src):
        warn = ""
    return Line(m, t, "; ".join(src) or "none", as_of, "; ".join(notes), warn)


def shifted(mu: Matchup, s_home: float, s_away: float) -> Matchup:
    return replace(mu, score_shift=(float(s_home), float(s_away)))


def ep_change(tm, mu: Matchup, side: int, shift: float, n: int = 4000) -> float:
    """Change in expected points per neutral-state drive (1st half, 15:00, tied,
    kickoff field position) implied by a scoring log-odds shift for one offense."""
    from .team_model import STATIC_COLS, dynamic_features
    rng = np.random.default_rng(0)
    yl = tm.fieldpos.sample_kick(rng, n)
    st = mu.home_static if side == 0 else mu.away_static
    X, _ = dynamic_features(np.asarray(st, float), STATIC_COLS, yl, np.ones(n, int),
                            np.full(n, 900.0), np.zeros(n))
    pts = np.zeros(tm.drive.W.shape[0])
    pts[OUT_IDX["TD"]] = 6 + tm.pat.xp_rate
    pts[OUT_IDX["FG"]] = 3
    pts[OUT_IDX["OPP_TD"]] = -(6 + tm.pat.xp_rate)
    pts[OUT_IDX["SAFETY"]] = -2
    off = np.zeros((n, len(pts)))
    off[:, OUT_IDX["TD"]] = shift
    off[:, OUT_IDX["FG"]] = shift
    return float((tm.drive.predict_proba(X, off) @ pts).mean() - (tm.drive.predict_proba(X) @ pts).mean())


@dataclass
class AnchorResult:
    sim: object
    d_home: float          # scoring log-odds shift, home offense
    d_away: float
    converged: bool
    iterations: int
    err_margin: float
    err_total: float
    base_margin: float     # unanchored mean margin with the same seed
    base_total: float


def solve_anchor(tm, mu: Matchup, target_margin: float, target_total: float, n_sims: int,
                 seed: int, tol: float = TOL, max_iter: int = 20, h: float = 0.1) -> AnchorResult:
    def run(d):
        r = simulate(tm, shifted(mu, d[0], d[1]), n_sims, seed=seed)
        return r, np.array([r.margin.mean() - target_margin, r.total.mean() - target_total])

    d = np.zeros(2)
    r, f = run(d)
    base = (float(r.margin.mean()), float(r.total.mean()))
    best = (np.abs(f).max(), r, d.copy(), f.copy())
    it = 0
    if np.abs(f).max() > tol:
        # finite-difference Jacobian
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2); e[j] = h
            _, fj = run(d + e)
            J[:, j] = (fj - f) / h
            it += 1
        while it < max_iter:
            try:
                step = -np.linalg.solve(J, f)
            except np.linalg.LinAlgError:
                step = -f * 0.01
            step = np.clip(step, -1.0, 1.0)
            d_new = d + step
            r_new, f_new = run(d_new)
            it += 1
            # Broyden update
            s = d_new - d
            if s @ s > 0:
                J = J + np.outer(f_new - f - J @ s, s) / (s @ s)
            d, f, r = d_new, f_new, r_new
            if np.abs(f).max() < best[0]:
                best = (np.abs(f).max(), r, d.copy(), f.copy())
            if np.abs(f).max() <= tol:
                break
    _, r, d, f = best
    return AnchorResult(r, float(d[0]), float(d[1]), bool(np.abs(f).max() <= tol), it,
                        float(f[0]), float(f[1]), base[0], base[1])


def describe(info: dict, home: str, away: str) -> list[str]:
    if info["mode"] != "market_anchored":
        out = [f"Mode: independent (model's own view of the game)."]
        if info.get("fallback"):
            out = [f"Mode: market_anchored requested but FELL BACK to independent: {info['fallback']}."]
        return out
    tm, tt = info["target_margin"], info["target_total"]
    return [
        f"Mode: market_anchored. Line: {home} {-tm:+g} (expected {home} margin {tm:+.1f}), total {tt:g}. "
        f"Source: {info['source']}." + (f" As of: {info['as_of']}." if info.get("as_of") else "")
        + (f" {info['warning']}" if info.get("warning") else ""),
        f"  Unanchored model: {home} margin {info['base_margin']:+.2f}, total {info['base_total']:.2f}.",
        f"  Offense adjustment (scoring log-odds; implied change in expected points per neutral drive): "
        f"{home} {info['d_home']:+.3f} ({info['ep_home']:+.2f} pts/drive), "
        f"{away} {info['d_away']:+.3f} ({info['ep_away']:+.2f} pts/drive). "
        f"Simulated after anchoring: margin {info['sim_margin']:+.2f}, total {info['sim_total']:.2f} "
        f"(errors {info['err_margin']:+.2f}, {info['err_total']:+.2f}; tolerance ±{TOL}; "
        f"{'converged' if info['converged'] else 'NOT converged'} in {info['iterations']} solver runs).",
    ]
