"""Before/after tables for the QB-exit and close-game changes.

Snapshots (data/processed/<dir>/ with the files the backtests write):
  before_qbexit  - before QB early exits
  after_qbexit   - QB exits added
  .              - current (QB exits + late-game score-situation states)
Usage: python scripts/cleanup_report.py [snapshot ...]
"""
import json
import sys

import numpy as np
import pandas as pd

from nflsim.ingest import PROCESSED

pd.set_option("display.width", 250)
snaps = sys.argv[1:] or ["before_qbexit", "after_qbexit", "."]


def rd(snap, name):
    p = PROCESSED / snap / name
    return json.loads(p.read_text()) if p.exists() else None


print("== QB1 (2024-25 player walk-forward, 1,000 sims/game) ==")
rows = []
for snap in snaps:
    for mode, sx in (("independent", ""), ("anchored", "_anchored")):
        r = rd(snap, f"player_wf_report{sx}.json")
        if r is None:
            continue
        for d in r["distributions"]:
            if d["role"] == "QB1" and d["stat"] in ("pass_att", "completions", "pass_yards"):
                rows.append({"snapshot": snap, "mode": mode, "stat": d["stat"], "n": d["n_player_games"],
                             "act_mean": d["act_mean"], "sim_mean": d["sim_mean"], "act_sd": d["act_sd"],
                             "sim_sd": d["sim_sd"], "act_p10": d["act_p10"], "sim_p10": d["sim_p10"],
                             "act_p90": d["act_p90"], "sim_p90": d["sim_p90"]})
print(pd.DataFrame(rows).round(2).to_string(index=False))

print("\n== Correlations (2024-25) ==")
rows = []
for snap in snaps:
    for mode, sx in (("independent", ""), ("anchored", "_anchored")):
        r = rd(snap, f"player_wf_report{sx}.json")
        if r is None:
            continue
        for c in r["correlations"]:
            rows.append({"pair": c["pair"], "col": f"{snap}|{mode}", "v": c["sim_corr_pooled"],
                         "actual": c["actual_corr"], "n": c["n_team_games"]})
c = pd.DataFrame(rows)
wide = c.pivot_table(index=["pair", "n", "actual"], columns="col", values="v").round(3)
print(wide.to_string())

print("\n== Known issues, 2021+ walk-forward (1,530 games, 4,000 sims) ==")
rows = []
for snap in snaps:
    for mode, sx in (("independent", ""), ("anchored", "_anchored")):
        r = rd(snap, f"wf_report{sx}.json")
        if r is None:
            continue
        e, L = r["evaluation"], r["league_validation"]
        pr = pd.read_parquet(PROCESSED / snap / f"wf_predictions{sx}.parquet")
        t = pr.home_score + pr.away_score
        row = {"snapshot": snap, "mode": mode,
               "total_bias": float((t - pr.sim_total).mean()),
               "margin=3": L["simulated"]["pct_margin_eq_3"], "<=7": L["simulated"]["pct_margin_le_7"],
               "OT": L["simulated"]["pct_overtime"], "ties": L["simulated"]["pct_ties"],
               "brier": e["wp"]["model_sim"]["brier"], "log_loss": e["wp"]["model_sim"]["log_loss"],
               "margin_mae": e["score"]["model_sim"]["margin_mae"], "total_mae": e["score"]["model_sim"]["total_mae"]}
        if "sim_qb_exit_home" in pr:
            row["sim_qb_exit_rate"] = float(np.r_[pr.sim_qb_exit_home, pr.sim_qb_exit_away].mean())
        rows.append(row)
        act = L["actual"]
print(pd.DataFrame(rows).round(4).to_string(index=False))
print("actual:", {k: round(act[k], 4) for k in ("pct_margin_eq_3", "pct_margin_le_7", "pct_overtime", "pct_ties")})
