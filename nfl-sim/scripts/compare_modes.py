"""Independent vs market_anchored: anchoring check, known issues, player validation.

Run after:
  python -m nflsim.backtest --sims 4000                               (independent)
  python -m nflsim.backtest --sims 4000 --mode market_anchored
  python -m nflsim.backtest --players --player-seasons 2024 2025 --sims 1000
  python -m nflsim.backtest --players --player-seasons 2024 2025 --sims 1000 --mode market_anchored
"""
import json

import numpy as np
import pandas as pd

from nflsim.ingest import PROCESSED

pd.set_option("display.width", 250)
ind = json.loads((PROCESSED / "wf_report.json").read_text())
anc = json.loads((PROCESSED / "wf_report_anchored.json").read_text())

print("== Anchoring check (2021+ walk-forward) ==")
print(json.dumps(anc["anchor_check"], indent=1, default=str))

print("\n== Known issues: simulated vs actual (same games) ==")
rows = []
for key in ["points_per_team_game", "total_mean", "total_sd", "margin_sd", "drives_per_game",
            "pct_margin_eq_3", "pct_margin_eq_7", "pct_margin_le_7", "pct_overtime", "pct_ties",
            "home_win_rate"]:
    rows.append({"metric": key, "actual": ind["league_validation"]["actual"][key],
                 "independent": ind["league_validation"]["simulated"][key],
                 "anchored": anc["league_validation"]["simulated"][key]})
print(pd.DataFrame(rows).round(4).to_string(index=False))
pi = pd.read_parquet(PROCESSED / "wf_predictions.parquet")
pa = pd.read_parquet(PROCESSED / "wf_predictions_anchored.parquet")
for name, p in (("independent", pi), ("anchored", pa)):
    t = p.home_score + p.away_score
    print(f"total bias (actual - simulated mean) {name}: {float((t - p.sim_total).mean()):+.3f}  by season: "
          + str((t - p.sim_total).groupby(p.season).mean().round(2).to_dict()))
print("market total bias (actual - closing total): "
      f"{float((pi.home_score + pi.away_score - pi.mkt_total).mean()):+.3f}")

print("\n== Game-level metrics: CHECK ONLY (anchored = market by construction, not skill) ==")
for name, r in (("independent", ind), ("anchored", anc)):
    e = r["evaluation"]
    print(f"{name:12s} Brier {e['wp']['model_sim']['brier']:.4f}  log loss {e['wp']['model_sim']['log_loss']:.4f}  "
          f"margin MAE {e['score']['model_sim']['margin_mae']:.3f}  total MAE {e['score']['model_sim']['total_mae']:.3f}"
          f"   (market: Brier {e['wp']['market_close']['brier']:.4f}, margin MAE {e['score']['market_close']['margin_mae']:.3f}, "
          f"total MAE {e['score']['market_close']['total_mae']:.3f})")

print("\n== Player validation 2024-25: independent vs anchored ==")
di = pd.DataFrame(json.loads((PROCESSED / "player_wf_report.json").read_text())["distributions"])
da = pd.DataFrame(json.loads((PROCESSED / "player_wf_report_anchored.json").read_text())["distributions"])
m = di.merge(da, on=["role", "stat"], suffixes=("_ind", "_anc"))
cols = ["role", "stat", "n_player_games_ind", "act_mean_ind", "sim_mean_ind", "sim_mean_anc", "act_sd_ind",
        "sim_sd_ind", "sim_sd_anc", "act_p10_ind", "sim_p10_ind", "sim_p10_anc", "act_p90_ind", "sim_p90_ind",
        "sim_p90_anc"]
print(m[cols].round(2).to_string(index=False))
ci = pd.DataFrame(json.loads((PROCESSED / "player_wf_report.json").read_text())["correlations"])
ca = pd.DataFrame(json.loads((PROCESSED / "player_wf_report_anchored.json").read_text())["correlations"])
c = ci.merge(ca, on="pair", suffixes=("_ind", "_anc"))
c["change"] = c["sim_corr_pooled_anc"] - c["sim_corr_pooled_ind"]
c["moved_gt_0.03"] = c["change"].abs() > 0.03
print(c[["pair", "n_team_games_ind", "actual_corr_ind", "sim_corr_pooled_ind", "sim_corr_pooled_anc", "change",
         "moved_gt_0.03"]].round(3).to_string(index=False))
