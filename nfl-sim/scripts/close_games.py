"""Diagnose close-game endings: actual vs simulated, 2018-2020 (tuning window).

  1. How 3-point final margins and overtimes arise (last scoring drive, who kicked,
     score before it, time).
  2. Late-game drive outcomes by exact score situation (offense view) and time left.
Simulations: weekly walk-forward fits, 200 sims per game with drive logs.
Usage: python scripts/close_games.py [n_sims]
"""
import sys

import numpy as np
import pandas as pd

from nflsim.backtest import load_inputs
from nflsim.simulator import matchup_from_features, simulate
from nflsim.team_model import fit_team_model

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
SEASONS = [2018, 2019, 2020]
games, gf, drives, pat = load_inputs()
test = games[games.season.isin(SEASONS) & games.label_home_score.notna()]


def situation(diff):
    return pd.cut(diff, [-99, -9, -4, -1, 0, 3, 8, 99],
                  labels=["down 9+", "down 4-8", "down 1-3", "tied", "up 1-3", "up 4-8", "up 9+"])


def timebin(secs):
    return pd.cut(secs, [-1, 120, 300, 600], labels=["<=2:00", "2-5 min", "5-10 min"])


# ---------------- actual drives (regulation + OT)
d = drives[drives.game_id.isin(test.game_id)].copy()
d["game_secs"] = np.where(d.half == 1, 1800 + d.half_seconds_remaining, d.half_seconds_remaining)
d["is_ot"] = d.half == 3
d["diff"] = d.score_differential
act = d[["game_id", "fixed_drive", "posteam", "is_ot", "game_secs", "diff", "result"]].rename(
    columns={"fixed_drive": "drive", "posteam": "offense", "result": "outcome"})
fin = test.set_index("game_id")
act["final_margin_abs"] = act.game_id.map((fin.label_home_score - fin.label_away_score).abs())
act["went_ot"] = act.game_id.map(fin.label_overtime == 1)

# ---------------- simulated drives
rows = []
rng = np.random.default_rng(0)
for (s, w), wk in test.groupby(["season", "week"], sort=True):
    tm = fit_team_model(wk.gameday.min(), drives, pat, gf)
    for g in wk.itertuples():
        r = simulate(tm, matchup_from_features(gf[gf.game_id == g.game_id].iloc[0]), N,
                     seed=int(rng.integers(1 << 31)), log_drives=True)
        L = r.drive_log
        own_after = np.where(L.offense == g.home_team, L.home_score, L.away_score)
        opp_after = np.where(L.offense == g.home_team, L.away_score, L.home_score)
        L = L.assign(diff=(own_after - L.off_pts) - (opp_after - L.def_pts),
                     game_secs=np.where(L.half == 1, 1800 + L.start_half_secs, L.start_half_secs),
                     is_ot=L.half == 3, game_id=g.game_id + "#" + L.sim.astype(str),
                     final_margin_abs=np.abs(r.margin)[L.sim], went_ot=r.overtime[L.sim])
        rows.append(L[["game_id", "drive", "offense", "is_ot", "game_secs", "diff", "outcome",
                       "final_margin_abs", "went_ot"]])
sim = pd.concat(rows, ignore_index=True)

pd.set_option("display.width", 220)
for name, x in (("ACTUAL", act), ("SIMULATED", sim)):
    ng = x.game_id.nunique()
    print(f"\n==================== {name}: {ng} games ====================")
    gm = x.groupby("game_id").agg(m=("final_margin_abs", "first"), ot=("went_ot", "first"))
    print(f"final margin = 3: {(gm.m == 3).mean():.3f}   <=7: {(gm.m <= 7).mean():.3f}   OT: {gm.ot.mean():.3f}")
    # last scoring drive of each game
    sc = x[x.outcome.isin(["TD", "FG", "SAFETY", "OPP_TD"])].sort_values(["game_id", "drive"])
    last = sc.groupby("game_id").tail(1).set_index("game_id").join(gm)
    m3 = last[last.m == 3]
    print(f"\n3-point finals (n={len(m3)}): how the last score happened")
    m3 = m3.assign(kind=np.where(m3.outcome == "FG", "FG", m3.outcome),
                   when=np.where(m3.is_ot, "OT", timebin(m3.game_secs).astype(str)),
                   before=situation(m3["diff"]).astype(str))
    print((pd.crosstab([m3.kind, m3.before], m3.when, margins=True) / len(m3)).round(3).to_string())
    # overtime: how the game got tied (last regulation scoring drive)
    reg = sc[~sc.is_ot]
    lastreg = reg.groupby("game_id").tail(1).set_index("game_id").join(gm)
    ot = lastreg[lastreg.ot]
    print(f"\nOT games (n={len(ot)}): last regulation score (tying score)")
    ot = ot.assign(when=timebin(ot.game_secs).astype(str).replace("nan", ">10 min"),
                   before=situation(ot["diff"]).astype(str))
    print((pd.crosstab([ot.outcome, ot.before], ot.when, margins=True) / max(len(ot), 1)).round(3).to_string())
    # late drive outcomes by exact situation (regulation, last 10 minutes)
    late = x[~x.is_ot & (x.game_secs <= 600)].assign(sit=lambda z: situation(z["diff"]),
                                                       t=lambda z: timebin(z.game_secs))
    late = late[late.sit.isin(["down 4-8", "down 1-3", "tied", "up 1-3", "up 4-8"])]
    tab = late.groupby(["t", "sit"], observed=True).outcome.value_counts(normalize=True).unstack(fill_value=0)
    tab["n_drives"] = late.groupby(["t", "sit"], observed=True).size()
    if name == "SIMULATED":
        tab["n_drives"] = (tab["n_drives"] / N).round(1)   # per real-game equivalents
    print("\nLate drives (last 10 min of regulation), outcome shares by time left x score situation"
          + (" (n per real-game equivalent)" if name == "SIMULATED" else ""))
    print(tab[["TD", "FG", "MISSED_FG", "PUNT", "TURNOVER", "DOWNS", "END_HALF", "n_drives"]].round(3).to_string())
