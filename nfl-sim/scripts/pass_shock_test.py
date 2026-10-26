"""Milestone 2.7, step 1, part A: is there a shared per-game passing shock?
Real data only (no simulation), 2018-2023.

The simulator picks each drive's play-by-play template independently, given the
drive's outcome, game state and start field-position bucket. So inside one
simulated team-game, passing residuals of different drives (relative to their
cell) are uncorrelated by construction. This script measures that correlation
in real games.

For each drive: dropbacks (targeted passes + throwaways/spikes + sacks), passing
yards, and yards per dropback. Residual = value - mean of its (outcome, state,
field-position bucket) cell, computed per season block so league trends drop out.
Pairwise residual correlations:
  same team-game          rho_game   (shock + persistent effects)
  same offense-season,
  different game          rho_off    (persistent offense effect, e.g. pass-heavy teams)
  same defense-season,
  different game          rho_def    (persistent defense effect)
Game-specific part ~ rho_game - rho_off - rho_def. 95% CIs bootstrap team-games
(2,000 resamples for rho_game; offense/defense-seasons for the others).

Usage: python scripts/pass_shock_test.py
"""
import json

import numpy as np
import pandas as pd

from nflsim.features import PROCESSED, read_processed
from nflsim.players import PASS, PASS_NT, SACK
from nflsim.team_model import start_bucket, state_index

SEASONS = range(2018, 2024)
rng = np.random.default_rng(0)

d = read_processed("drives")
d = d[d["season"].isin(SEASONS) & (d["season_type"] == "REG")].copy()
p = pd.read_parquet(PROCESSED / "drive_plays.parquet")
p = p[p["drive_uid"].isin(d["drive_uid"])]
db = np.isin(p["kind"], [PASS, PASS_NT, SACK])
agg = pd.DataFrame({"drive_uid": p["drive_uid"], "db": db.astype(int),
                    "pyd": np.where((p["kind"] == PASS) & (p["complete"] == 1), p["yards"], 0.0)}
                   ).groupby("drive_uid").sum()
d = d.merge(agg, left_on="drive_uid", right_index=True, how="inner")
d["ypd"] = np.where(d["db"] > 0, d["pyd"] / d["db"].clip(lower=1), np.nan)
d["state"] = state_index(d["half"].to_numpy(), d["half_seconds_remaining"].to_numpy(),
                         d["score_differential"].to_numpy())
d["bucket"] = start_bucket(d["yardline_100"].to_numpy())
d["cell"] = d["result"].astype(str) + "|" + d["state"].astype(str) + "|" + d["bucket"].astype(str)
d["tg"] = d["game_id"] + "|" + d["posteam"]
d["os"] = d["posteam"] + "|" + d["season"].astype(str)
d["ds"] = d["defteam"] + "|" + d["season"].astype(str)


def residual(col):
    x = d[col]
    m = x.groupby([d["cell"], d["season"]]).transform("mean")
    n = x.groupby([d["cell"], d["season"]]).transform("count")
    return (x - m).where(n >= 20)   # skip tiny cells


def pair_cov(r, outer, inner=None):
    """Sum over pairs of drives sharing `outer` (and, if inner given, in different
    `inner` groups) of r_i r_j, plus the number of such pairs. Per outer group."""
    df = pd.DataFrame({"r": r, "o": outer, "i": inner if inner is not None else 0}).dropna()
    so = df.groupby("o")["r"].agg(["sum", "count"])
    sq = (df["r"] ** 2).groupby(df["o"]).sum()
    s_pairs = so["sum"] ** 2 - sq
    n_pairs = so["count"] * (so["count"] - 1)
    if inner is not None:
        si = df.groupby(["o", "i"])["r"].agg(["sum", "count"])
        si2 = (df["r"] ** 2).groupby([df["o"], df["i"]]).sum()
        within = (si["sum"] ** 2 - si2).groupby(level=0).sum()
        nwithin = (si["count"] * (si["count"] - 1)).groupby(level=0).sum()
        s_pairs, n_pairs = s_pairs - within, n_pairs - nwithin
    return pd.DataFrame({"s": s_pairs, "n": n_pairs})


def rho(pc, var, idx=None):
    x = pc if idx is None else pc.iloc[idx]
    return float(x["s"].sum() / x["n"].sum() / var)


out = {}
for col, label in (("db", "dropbacks per drive"), ("pyd", "passing yards per drive"),
                   ("ypd", "yards per dropback (drives with a dropback)")):
    r = residual(col)
    var = float(np.nanmean(r ** 2))
    res = {"n_drives": int(r.notna().sum()), "resid_sd": var ** 0.5}
    for name, outer, inner in (("rho_game", d["tg"], None), ("rho_off", d["os"], d["game_id"]),
                               ("rho_def", d["ds"], d["game_id"])):
        pc = pair_cov(r, outer, inner)
        est = rho(pc, var)
        bs = [rho(pc, var, rng.integers(0, len(pc), len(pc))) for _ in range(2000 if inner is None else 1000)]
        res[name] = est
        res[name + "_ci95"] = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
    res["rho_game_specific"] = res["rho_game"] - res["rho_off"] - res["rho_def"]
    nper = r.notna().groupby(d["tg"]).sum()
    nbar = float(nper.mean())
    res["drives_per_team_game"] = nbar
    # Variance of a team-game sum relative to independent drives: 1 + (n-1) rho
    res["team_game_var_inflation_total"] = 1 + (nbar - 1) * res["rho_game"]
    res["team_game_var_inflation_game_specific"] = 1 + (nbar - 1) * res["rho_game_specific"]
    out[label] = res

print(json.dumps(out, indent=1))
(PROCESSED / "pass_shock_test_drives.json").write_text(json.dumps(out, indent=1))
