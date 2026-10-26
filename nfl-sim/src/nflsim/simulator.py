"""Drive-level NFL game simulator, vectorized across simulations.

Each simulation alternates possessions from an opening kickoff. For each drive
we sample, from the fitted TeamModel:
  outcome   ~ DriveModel(matchup, field position, clock, score state)
  plays, seconds, pass share ~ PaceTable(outcome, game state)
  next start yard line       ~ FieldPosTable(outcome, start yard line)
  PAT / two-point            ~ PATModel(score state)
The number of drives is therefore random: it is whatever fits in the clock
given the sampled drive durations (which depend on game state).

Score and clock are tracked by quarter. Overtime follows the rules of the
season (10-minute regular-season OT since 2017; guaranteed possession in
playoffs from 2022 and regular season from 2025; regular-season ties allowed).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .team_model import (OUT_IDX, OUTCOMES, STATIC_COLS, TeamModel, dynamic_features)

TD, FG, MFG, PUNT, TO, DOWNS, EOH, SAF, OPPTD = (OUT_IDX[o] for o in OUTCOMES)


@dataclass
class Matchup:
    home_team: str
    away_team: str
    home_static: np.ndarray   # STATIC_COLS for home offense vs away defense
    away_static: np.ndarray   # STATIC_COLS for away offense vs home defense
    season: int
    playoff: bool = False
    # Market anchoring: additive log-odds on the offense's scoring outcomes
    # (TD and FG) in the drive-outcome model, per offense (home, away). 0 = off.
    score_shift: tuple = (0.0, 0.0)
    # STATIC_COLS with the backup QB in; None disables starter exits for that team.
    home_static_backup: np.ndarray | None = None
    away_static_backup: np.ndarray | None = None
    qb_exits: bool = True
    # Per-game passing shock (Milestone 2.7): each team gets z ~ N(0, 1) per
    # simulation, and every drive's template is drawn with pass-share tilt
    # pass_shock * z. 0 = off.
    pass_shock: float = 0.0


@dataclass
class SimResult:
    home_score: np.ndarray
    away_score: np.ndarray
    home_q: np.ndarray        # [n, 5] points by quarter (index 4 = OT)
    away_q: np.ndarray
    n_drives: np.ndarray      # total drives (both teams)
    home_drives: np.ndarray
    overtime: np.ndarray
    drive_log: pd.DataFrame | None
    drives: dict | None = None   # flat per-drive arrays, in simulation order

    @property
    def margin(self):
        return self.home_score - self.away_score

    @property
    def total(self):
        return self.home_score + self.away_score

    def home_win_prob(self) -> float:
        m = self.margin
        return float(((m > 0).sum() + 0.5 * (m == 0).sum()) / len(m))


def matchup_from_features(gf_row: pd.Series) -> Matchup:
    from .team_model import team_side_features
    sides = team_side_features(gf_row.to_frame().T.reset_index(drop=True))
    backup = {}
    for side in ("home", "away"):
        x = sides[side]
        # starter exits are only simulated when the depth chart names a backup
        if "qb_epa_b" in x and np.isfinite(x["qb_epa_b"].iloc[0]) and \
                isinstance(gf_row.get(f"{side}_qb2_id"), str):
            b = x[STATIC_COLS].copy()
            b["qb_epa"], b["qb_delta"] = x["qb_epa_b"], x["qb_delta_b"]
            backup[side] = b.to_numpy(float)[0]
    return Matchup(
        home_team=gf_row["home_team"], away_team=gf_row["away_team"],
        home_static=sides["home"][STATIC_COLS].to_numpy(float)[0],
        away_static=sides["away"][STATIC_COLS].to_numpy(float)[0],
        season=int(gf_row["season"]), playoff=gf_row.get("game_type", "REG") != "REG",
        home_static_backup=backup.get("home"), away_static_backup=backup.get("away"),
    )


# Size (sd of the pass-share tilt) used when the per-game passing shock is on
# (--pass-shock on). Estimated on 2018-2023 (README, Milestone 2.7). The switch
# defaults to off: the shock failed the keep test (wider player spreads than
# actual, worse 2021+ total error).
PASS_SHOCK_SD = 0.4


# Starting-QB exit hazard per drive, by state at the start of the drive.
# Estimated on 2018-2020 play-by-play (scripts/qb_exits.py): 217 permanent exits
# in 1,606 team-games (124 blowout, 93 injury/benching). Blowouts are split by
# whether the offense leads (leaders pull starters ~3x as often) and non-blowout
# (injury / benching) hazards by quarter (exits skew late). (exits / drives)
QB_EXIT_NORMAL_BY_Q = {1: 14 / 4288, 2: 12 / 4590, 3: 30 / 3953, 4: 34 / 3311}
QB_EXIT_BLOWOUT = [   # (min |margin|, max secs left, min secs left (excl.), hazard leading, trailing)
    (17, 900, 300, 8 / 157, 8 / 206),
    (17, 300, -1, 20 / 98, 8 / 100),
    (24, 1800, 900, 5 / 106, 9 / 156),
    (24, 900, 300, 22 / 106, 10 / 143),
    (24, 300, -1, 26 / 51, 10 / 67),
]


def qb_exit_is_blowout(game_secs: np.ndarray, diff: np.ndarray) -> np.ndarray:
    ad = np.abs(diff)
    return ((ad >= 17) & (game_secs <= 900)) | ((ad >= 24) & (game_secs <= 1800))


def qb_exit_hazard(game_secs: np.ndarray, diff: np.ndarray) -> np.ndarray:
    """diff = offense score minus defense score."""
    ad = np.abs(diff)
    q = np.select([game_secs > 2700, game_secs > 1800, game_secs > 900], [1, 2, 3], 4)
    h = np.select([q == 1, q == 2, q == 3], [QB_EXIT_NORMAL_BY_Q[1], QB_EXIT_NORMAL_BY_Q[2],
                                             QB_EXIT_NORMAL_BY_Q[3]], QB_EXIT_NORMAL_BY_Q[4])
    for m, hi, lo, hz_lead, hz_trail in QB_EXIT_BLOWOUT:   # later rows (24+) override 17+
        h = np.where((ad >= m) & (game_secs <= hi) & (game_secs > lo),
                     np.where(diff > 0, hz_lead, hz_trail), h)
    return h


def simulate(tm: TeamModel, mu: Matchup, n_sims: int = 2000, seed: int | None = None,
             log_drives: bool = False, max_drives: int = 80) -> SimResult:
    rng = np.random.default_rng(seed)
    n = n_sims
    score = np.zeros((n, 2))
    qpts = np.zeros((n, 2, 5))
    ndr = np.zeros((n, 2), int)
    half = np.ones(n, int)
    half_secs = np.full(n, 1800.0)
    first_recv = rng.integers(0, 2, n)          # coin toss: who receives opening kick
    pos = first_recv.copy()                      # 0 = home offense, 1 = away offense
    yl = tm.fieldpos.sample_kick(rng, n)
    done = np.zeros(n, bool)
    ot = np.zeros(n, bool)
    ot_drives = np.zeros(n, int)
    ot_first = np.full(n, -1)
    ot_len = 900.0 if (mu.season < 2017 or mu.playoff) else 600.0
    guaranteed = (mu.playoff and mu.season >= 2022) or mu.season >= 2025
    static = np.vstack([mu.home_static, mu.away_static])
    has_b = np.array([mu.home_static_backup is not None, mu.away_static_backup is not None]) & mu.qb_exits
    static_b = np.vstack([mu.home_static_backup if has_b[0] else mu.home_static,
                          mu.away_static_backup if has_b[1] else mu.away_static])
    qb_out = np.zeros((n, 2), bool)   # starter has left the game (permanent)
    qb_blow = np.zeros((n, 2), bool)  # ...and it was a blowout pull (other starters sit too)
    # Shared passing shock per (simulation, team), from its own stream so the
    # main stream (and so the simulation with the shock off) is unchanged.
    shock = (mu.pass_shock * np.random.default_rng(None if seed is None else [seed, 7])
             .standard_normal((n, 2))) if mu.pass_shock else None
    log = []
    drv = []    # per-step (sims, offense side, outcome, start yl, template uid, half)

    for step in range(max_drives):
        a = np.flatnonzero(~done)
        if a.size == 0:
            break
        # Common random numbers: every simulation gets the same uniforms at the
        # same drive step regardless of what other simulations are doing, so a
        # small change to the model (market anchoring) only changes outcomes whose
        # uniform crosses a threshold. Columns: outcome, pace, field position,
        # PAT decision, PAT success, half-time kick, OT coin toss, OT kick, QB exit.
        U = rng.random((n, 9))
        p = pos[a]
        o = 1 - p
        diff = score[a, p] - score[a, o]
        h = np.where(ot[a], 3, half[a])
        # Starter exit before this drive (blowout or injury/benching hazard).
        if has_b.any():
            gsecs = np.where(h == 1, 1800 + half_secs[a], half_secs[a])
            leave = (~qb_out[a, p]) & has_b[p] & (U[a, 8] < qb_exit_hazard(gsecs, diff))
            qb_out[a[leave], p[leave]] = True
            blow = leave & qb_exit_is_blowout(gsecs, diff)
            qb_blow[a[blow], p[blow]] = True
        cur_out = qb_out[a, p]
        cur_blow = qb_blow[a, p]
        X, st = dynamic_features(np.where(cur_out[:, None], static_b[p], static[p]), STATIC_COLS,
                                 yl[a], h, half_secs[a], diff)
        if mu.score_shift != (0.0, 0.0):
            off = np.zeros((a.size, len(OUTCOMES)))
            sh = np.asarray(mu.score_shift, float)[p]
            off[:, TD] = sh
            off[:, FG] = sh
            P = tm.drive.predict_proba(X, off)
        else:
            P = tm.drive.predict_proba(X)
        u = U[a, 0]
        out = (P.cumsum(1) < u[:, None]).sum(1)
        out = np.minimum(out, len(OUTCOMES) - 1)
        pace, tmpl = tm.pace.sample(rng, out, st, yl[a], U[a, 1],
                                    tilt=None if shock is None else shock[a, p])
        plays, secs, npr, npass = pace.T
        secs = np.where(out == EOH, half_secs[a], np.minimum(secs, half_secs[a]))
        start_yl = yl[a].copy()
        start_secs = half_secs[a].copy()

        # Scoring
        late = (h >= 2) & ((half_secs[a] - secs <= 900) | (h == 3))
        pts_off = np.zeros(a.size)
        pts_def = np.zeros(a.size)
        m = out == TD
        if m.any():
            pts_off[m] = 6 + tm.pat.points(rng, late[m], diff[m] + 6, U[a[m]][:, 3:5])
        pts_off[out == FG] = 3
        m = out == OPPTD
        if m.any():
            pts_def[m] = 6 + tm.pat.points(rng, late[m], -diff[m] + 6, U[a[m]][:, 3:5])
        pts_def[out == SAF] = 2
        score[a, p] += pts_off
        score[a, o] += pts_def

        # Clock and quarter of the scoring event (end of drive)
        half_secs[a] -= secs
        elapsed = np.where(h == 1, 1800 - half_secs[a], 3600 - half_secs[a])
        qi = np.where(h == 3, 4, np.clip((elapsed // 900).astype(int), 0, 3))
        qi = np.where((half_secs[a] <= 0) & (h <= 2), np.where(h == 1, 1, 3), qi)
        qpts[a, p, qi] += pts_off
        qpts[a, o, qi] += pts_def
        ndr[a, p] += 1
        drv.append((a, p, out, start_yl, tmpl, h, cur_out, cur_blow))

        if log_drives:
            log.append(pd.DataFrame({
                "sim": a, "drive": ndr[a].sum(1), "offense": np.where(p == 0, mu.home_team, mu.away_team),
                "half": h,
                "quarter": np.where(h == 3, 5, (h - 1) * 2 + np.where(start_secs > 900, 1, 2)),
                "start_half_secs": start_secs, "start_yl": start_yl,
                "outcome": np.array(OUTCOMES)[out], "plays": plays, "seconds": secs,
                "pass_share": np.where(npr > 0, npass / np.maximum(npr, 1), np.nan),
                "state": st, "off_pts": pts_off, "def_pts": pts_def,
                "home_score": score[a, 0], "away_score": score[a, 1],
            }))

        # Next possession and field position
        same_team = out == OPPTD           # defense scored, then kicks off to the offense
        new_pos = np.where(same_team, p, o)
        new_yl = tm.fieldpos.sample(rng, out, start_yl, U[a, 2])
        pos[a] = new_pos
        yl[a] = new_yl

        # Overtime termination
        if ot[a].any():
            mo = ot[a]
            ai = a[mo]
            ot_drives[ai] += 1
            ot_first[ai] = np.where(ot_drives[ai] == 1, out[mo], ot_first[ai])
            d_now = score[ai, 0] != score[ai, 1]
            if guaranteed:
                ends = d_now & ((ot_drives[ai] >= 2) | np.isin(ot_first[ai], [OPPTD, SAF]))
            else:
                ends = d_now & ((ot_drives[ai] >= 2) | (ot_first[ai] != FG))
            done[ai[ends]] = True

        # End of half / regulation / OT period
        expired = a[(half_secs[a] <= 0) & ~done[a]]
        if expired.size:
            h1 = expired[(half[expired] == 1) & ~ot[expired]]
            half[h1] = 2
            half_secs[h1] = 1800.0
            pos[h1] = 1 - first_recv[h1]
            yl[h1] = tm.fieldpos.sample_kick(rng, h1.size, U[h1, 5])
            rest = expired[~np.isin(expired, h1)]
            tied = score[rest, 0] == score[rest, 1]
            done[rest[~tied]] = True
            go_ot = rest[tied & (~ot[rest] | mu.playoff)]
            if not mu.playoff:
                done[rest[tied & ot[rest]]] = True   # regular-season tie
            ot[go_ot] = True
            half_secs[go_ot] = ot_len
            new = go_ot[ot_drives[go_ot] == 0]
            pos[new] = (U[new, 6] < 0.5).astype(int)
            yl[go_ot] = tm.fieldpos.sample_kick(rng, go_ot.size, U[go_ot, 7])
    done[:] = True

    return SimResult(
        home_score=score[:, 0].astype(int), away_score=score[:, 1].astype(int),
        home_q=qpts[:, 0].astype(int), away_q=qpts[:, 1].astype(int),
        n_drives=ndr.sum(1), home_drives=ndr[:, 0], overtime=ot,
        drive_log=pd.concat(log, ignore_index=True) if log else None,
        drives={k: np.concatenate([x[i] for x in drv]) for i, k in
                enumerate(["sim", "side", "outcome", "start_yl", "uid", "half", "qb_out",
                           "qb_blowout"])},
    )


def main(argv=None):
    """Simulate one game: python -m nflsim.simulator --game-id 2024_01_BAL_KC"""
    import argparse

    from .backtest import load_inputs
    from .team_model import expected_points_per_drive, fit_team_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id")
    ap.add_argument("--home")
    ap.add_argument("--away")
    ap.add_argument("--season", type=int)
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    games, gf, drives, pat = load_inputs()
    if a.game_id:
        row = gf[gf["game_id"] == a.game_id]
    else:
        row = gf[(gf["home_team"] == a.home) & (gf["away_team"] == a.away)]
        if a.season:
            row = row[row["season"] == a.season]
        row = row.sort_values("gameday").tail(1)
    if row.empty:
        raise SystemExit("game not found in schedule")
    row = row.iloc[0]
    tm = fit_team_model(row["gameday"], drives, pat, gf)
    mu = matchup_from_features(row)
    r = simulate(tm, mu, a.sims, seed=a.seed, log_drives=True)
    print(f"{row['game_id']}  {row['gameday']:%Y-%m-%d}  model trained on {tm.n_train_drives} "
          f"drives before {tm.cutoff:%Y-%m-%d}; features as of {row['asof_date']:%Y-%m-%d}")
    print(f"Starting QBs (pregame estimate): {row['home_team']} {row['home_qb_id']} "
          f"[{row['home_qb_source']}], {row['away_team']} {row['away_qb_id']} [{row['away_qb_source']}]")
    print("Weather: unavailable (not used)")
    for side, st in (("home", mu.home_static), ("away", mu.away_static)):
        ep = expected_points_per_drive(tm, st)
        print(f"  {row[side + '_team']} offense: EP/drive {ep['ep_per_drive']:.2f}  "
              f"TD {ep['TD']:.3f} FG {ep['FG']:.3f} PUNT {ep['PUNT']:.3f} "
              f"TO {ep['TURNOVER'] + ep['DOWNS']:.3f}")
    wp = r.home_win_prob()
    se = (wp * (1 - wp) / a.sims) ** 0.5
    print(f"P({row['home_team']} win) = {wp:.3f}  (Monte Carlo SE {se:.4f}; model error not included)")
    print(f"Mean score {row['home_team']} {r.home_score.mean():.1f} - {row['away_team']} "
          f"{r.away_score.mean():.1f}; margin sd {r.margin.std():.1f}; total {r.total.mean():.1f}")
    print(f"Drives/game {r.n_drives.mean():.1f}; OT {r.overtime.mean():.3f}; ties "
          f"{(r.margin == 0).mean():.4f}")
    print("\nOne simulated game (sim 0):")
    lg = r.drive_log[r.drive_log["sim"] == 0]
    print(lg[["drive", "quarter", "offense", "start_half_secs", "start_yl", "outcome", "plays",
              "seconds", "home_score", "away_score"]].to_string(index=False))
    print(f"Final: {row['home_team']} {r.home_score[0]} - {row['away_team']} {r.away_score[0]}  "
          f"by quarter (Q1-Q4, OT) {r.home_q[0].tolist()} / {r.away_q[0].tolist()}")


if __name__ == "__main__":
    main()
