"""Walk-forward on the 2018-2020 tuning window only; saves predictions + samples.
Usage: python scripts/tune_window.py <tag> [sims] [pass_shock_sd]"""
import json
import sys

from nflsim.backtest import evaluate, league_validation, walk_forward
from nflsim.features import PROCESSED, read_processed

tag = sys.argv[1]
sims = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
shock = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
pr, sm = walk_forward(2018, sims, verbose=False, end_season=2020, save=False, pass_shock=shock)
pr.to_parquet(PROCESSED / f"tune_{tag}_pred.parquet", index=False)
sm.to_parquet(PROCESSED / f"tune_{tag}_samples.parquet", index=False)
e = evaluate(pr)
L = league_validation(pr, sm, read_processed("drives"))
out = {"brier": e["wp"]["model_sim"]["brier"], "log_loss": e["wp"]["model_sim"]["log_loss"],
       "margin_mae": e["score"]["model_sim"]["margin_mae"], "total_mae": e["score"]["model_sim"]["total_mae"],
       "total_bias": e["score"]["model_sim"]["total_bias"], "n": e["n_games"],
       **{k: L["simulated"][k] for k in ("pct_margin_eq_3", "pct_margin_le_7", "pct_overtime", "pct_ties",
                                         "drives_per_game", "total_mean")},
       "actual": {k: L["actual"][k] for k in ("pct_margin_eq_3", "pct_margin_le_7", "pct_overtime",
                                              "pct_ties", "drives_per_game", "total_mean")}}
(PROCESSED / f"tune_{tag}.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
