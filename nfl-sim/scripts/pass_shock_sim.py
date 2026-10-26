"""Milestone 2.7, step 1, part B: simulated vs actual game-to-game variation of
team passing, 2018-2023, and whether the excess explains the QB-receiver gap.

Needs the 2018-2023 player walk-forward (current model, no changes):
  python -m nflsim.backtest --players --player-seasons 2018 2019 2020 2021 2022 2023 --sims 1000 --tag _1823
Usage: python scripts/pass_shock_sim.py [file suffix, default _1823; e.g. _shock_1823_k0.3]
(section 3 is meaningful only for a run with the shock off)

1. Team-game passing (all QBs): pass attempts, passing yards, yards per attempt,
   pass rate (attempts / (attempts + designed runs)), drives. For each:
     predictive variance  = mean over team-games of the within-game simulated variance
     realized variance    = mean of (actual - simulated mean)^2, minus the Monte Carlo
                            variance of that simulated mean (sim var / 100 kept sims)
   If the simulator's game-to-game spread were right, the ratio would be ~1.
   Excess = realized - predictive (it also contains pregame mean error).
2. QB-receiver correlations in the same games (same method as the M2 report).
3. Counterfactual (post hoc, no model change): multiply every passing stat of a
   team in one simulation (QB yards, each receiver's yards) by a shared
   m = exp(s z - s^2/2), z ~ N(0,1), with s set so the simulated pass-attempt
   variance gains exactly the measured excess. Then recompute the correlations and
   player sds. This shows how much of the gap a shared volume shock of the measured
   size could close, and what it does to player spreads.
"""
import json
import sys

import numpy as np
import pandas as pd

from nflsim.backtest import correlation_report
from nflsim.features import PROCESSED, read_processed
from nflsim.players import PASS, PASS_NT, RUN

TAG = sys.argv[1] if len(sys.argv) > 1 else "_1823"
gc = pd.read_pickle(PROCESSED / f"player_wf_gamecorr{TAG}.pkl")
pr = pd.read_parquet(PROCESSED / f"player_wf{TAG}.parquet")
rng = np.random.default_rng(0)

# Actual team-game passing from play-by-play (same play definitions as the sim)
d = read_processed("drives")
d = d[d["season"].between(2018, 2023)]
p = pd.read_parquet(PROCESSED / "drive_plays.parquet")
p = p.merge(d[["drive_uid", "game_id", "posteam"]], on="drive_uid")
act = pd.DataFrame({
    "game_id": p["game_id"], "team": p["posteam"],
    "pass_att": np.isin(p["kind"], [PASS, PASS_NT]).astype(int),
    "pass_yards": np.where((p["kind"] == PASS) & (p["complete"] == 1), p["yards"], 0.0),
    "designed_runs": (p["kind"] == RUN).astype(int)}).groupby(["game_id", "team"]).sum()
act["drives"] = d.groupby(["game_id", "posteam"]).size().rename_axis(["game_id", "team"])

rows = []
for rec in gc:
    key = (rec["game_id"], rec["team"])
    if key not in act.index:
        continue
    a = act.loc[key]
    s_att, s_yd = rec["team_pass_att"].astype(float), rec["team_pass_yards"].astype(float)
    s_run, s_dr = rec["team_designed_runs"].astype(float), rec["team_drives"].astype(float)
    sims = {"pass_att": s_att, "pass_yards": s_yd,
            "ypa": np.where(s_att > 0, s_yd / np.maximum(s_att, 1), np.nan),
            "pass_rate": s_att / np.maximum(s_att + s_run, 1), "drives": s_dr}
    acts = {"pass_att": a["pass_att"], "pass_yards": a["pass_yards"],
            "ypa": a["pass_yards"] / a["pass_att"] if a["pass_att"] > 0 else np.nan,
            "pass_rate": a["pass_att"] / max(a["pass_att"] + a["designed_runs"], 1), "drives": a["drives"]}
    r = {"game_id": rec["game_id"], "team": rec["team"]}
    for k, x in sims.items():
        r[k + "_act"] = acts[k]
        r[k + "_mean"] = np.nanmean(x)
        r[k + "_var"] = np.nanvar(x, ddof=1)
        r[k + "_n"] = int(np.isfinite(x).sum())
    rows.append(r)
tg = pd.DataFrame(rows)

out = {"n_team_games": int(len(tg)), "variance": {}}
for k in ("pass_att", "pass_yards", "ypa", "pass_rate", "drives"):
    x = tg.dropna(subset=[k + "_act", k + "_mean"])
    pred = float(x[k + "_var"].mean())
    real = float(((x[k + "_act"] - x[k + "_mean"]) ** 2).mean() - (x[k + "_var"] / x[k + "_n"]).mean())
    # bootstrap CI of the ratio over team-games
    e2 = ((x[k + "_act"] - x[k + "_mean"]) ** 2 - x[k + "_var"] / x[k + "_n"]).to_numpy()
    v = x[k + "_var"].to_numpy()
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(x), len(x))
        bs.append(e2[i].mean() / v[i].mean())
    out["variance"][k] = {
        "n": int(len(x)), "actual_mean": float(x[k + "_act"].mean()), "sim_mean": float(x[k + "_mean"].mean()),
        "actual_sd_pooled": float(x[k + "_act"].std()),
        "predictive_sd": pred ** 0.5, "realized_sd": max(real, 0) ** 0.5,
        "ratio_realized_to_predictive_var": real / pred,
        "ratio_ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
        "excess_var": real - pred}

# Correlations, current model
base = correlation_report(gc).set_index("pair")
out["correlations_current"] = base.reset_index().to_dict("records")


def shocked(gc, s, stats_mult):
    """Copy of gamecorr with a shared multiplicative shock on passing stats."""
    new = []
    for rec in gc:
        r = dict(rec)
        m = np.exp(s * rng.standard_normal(len(rec["sim_total"])) - s * s / 2)
        for k in list(rec):
            if not k.startswith("act_") and any(k.endswith(t) for t in stats_mult) and \
                    isinstance(rec[k], np.ndarray):
                r[k] = rec[k] * m
        new.append(r)
    return new


v = out["variance"]["pass_att"]
x = tg.dropna(subset=["pass_att_act"])
second = float((x["pass_att_mean"] ** 2 + x["pass_att_var"]).mean())
s2 = max(v["excess_var"], 0) / second
s = s2 ** 0.5
out["counterfactual_shock_sd_logscale"] = s
gcs = shocked(gc, s, ("_pass_yards", "_rec_yards", "_pass_att", "_targets", "_receptions"))
cf = correlation_report(gcs).set_index("pair")
cmp_ = base[["n_team_games", "actual_corr", "sim_corr_pooled"]].join(
    cf[["sim_corr_pooled"]].rename(columns={"sim_corr_pooled": "sim_corr_with_shock"}))
cmp_["gap_now"] = cmp_["sim_corr_pooled"] - cmp_["actual_corr"]
cmp_["gap_with_shock"] = cmp_["sim_corr_with_shock"] - cmp_["actual_corr"]
out["correlations_counterfactual"] = cmp_.reset_index().to_dict("records")


def sds(g, act_pr):
    res = {}
    for role, st in (("QB1", "pass_yards"), ("QB1", "pass_att"), ("WR1", "rec_yards"), ("WR2", "rec_yards"),
                     ("TE1", "rec_yards"), ("WR1", "targets")):
        sim = np.concatenate([r[f"{role}_{st}"] for r in g if f"{role}_{st}" in r
                              and np.isfinite(r.get(f"act_{role}_{st}", np.nan))])
        sim = sim[np.isfinite(sim)]
        a = act_pr[(act_pr["role"] == role) & act_pr["played"]]["act_" + st]
        res[f"{role} {st}"] = {"actual_sd": float(a.std()), "sim_sd": float(sim.std())}
    return res


s0, s1 = sds(gc, pr), sds(gcs, pr)
out["player_sd"] = {k: {**s0[k], "sim_sd_with_shock": s1[k]["sim_sd"]} for k in s0}

pd.set_option("display.width", 220)
print(pd.DataFrame(out["variance"]).T.round(3).to_string())
print(f"\ncounterfactual shock sd (log scale): {s:.3f}")
print(cmp_.round(3).to_string())
print(pd.DataFrame(out["player_sd"]).T.round(1).to_string())
(PROCESSED / f"pass_shock_test_sim{TAG}.json").write_text(json.dumps(out, indent=1, default=float))
