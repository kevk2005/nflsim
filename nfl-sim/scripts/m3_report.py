"""Milestone 3 report: compare the four prop/SGP backtest configurations and keep
the best-calibrated one (pass shock off everywhere).

Run after (each writes data/processed/prop_wf_report<suffix>.json):
  python -m nflsim.backtest --props --mode independent     --qb-exits on|off
  python -m nflsim.backtest --props --mode market_anchored --qb-exits on|off

Selection rule (fixed before looking at results): lowest mean of the props log
loss (self-set lines) and the synthetic-SGP K/N log loss. The choice is saved to
data/processed/prop_wf_best.json; the app uses it as its default mode/QB exits.
"""
import json

import pandas as pd

from nflsim.ingest import PROCESSED

CONFIGS = [("independent", "on", ""), ("independent", "off", "_noexits"),
           ("market_anchored", "on", "_anchored"), ("market_anchored", "off", "_anchored_noexits")]
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

reps = {}
for m, q, sx in CONFIGS:
    p = PROCESSED / f"prop_wf_report{sx}.json"
    if p.exists():
        reps[(m, q)] = json.loads(p.read_text())
rows = []
for (m, q), r in reps.items():
    po, sj, si = r["props_overall"], r["sgp_overall"]["joint"], r["sgp_overall"]["independence"]
    rows.append({"mode": m, "qb_exits": q, "props_n": po["n"], "props_brier": po["brier"],
                 "props_logloss": po["log_loss"], "props_mean_pred": po["mean_pred"],
                 "props_actual": po["actual_rate"], "sgp_n": sj["n"], "sgp_brier": sj["brier"],
                 "sgp_logloss": sj["log_loss"], "sgp_mean_pred": sj["mean_pred"],
                 "sgp_actual": sj["actual_rate"], "indep_logloss": si["log_loss"],
                 "indep_mean_pred": si["mean_pred"]})
cmp_ = pd.DataFrame(rows)
cmp_["score"] = (cmp_["props_logloss"] + cmp_["sgp_logloss"]) / 2
print("== Configurations ==")
print(cmp_.round(4).to_string(index=False))
best = cmp_.loc[cmp_["score"].idxmin()]
out = {"mode": best["mode"], "qb_exits": best["qb_exits"], "score": float(best["score"]),
       "rule": "lowest mean of props log loss and SGP K/N log loss",
       "source": f"scripts/m3_report.py over {len(cmp_)} configs"}
(PROCESSED / "prop_wf_best.json").write_text(json.dumps(out, indent=1))
print(f"\nBest: {out['mode']} / QB exits {out['qb_exits']}  (score {out['score']:.4f})")

r = reps[(best["mode"], best["qb_exits"])]
print("\n== Props by market (best config) ==")
pm = pd.DataFrame({k: {kk: v[kk] for kk in ("n", "mean_pred", "actual_rate", "brier",
                                             "brier_insample_base_rate", "log_loss")}
                   for k, v in r["props_by_market"].items()}).T
print(pm.round(4).to_string())
print("\n== Props by season ==")
print(pd.DataFrame(r["props_by_season"]).T.round(4).to_string())
print("\n== Props calibration (all) ==")
print(pd.DataFrame(r["props_calibration"]).round(3).to_string(index=False))
print("\n== SGP by template ==")
print(pd.DataFrame(r["sgp_by_template"]).T.drop(columns=["actual_ci95"]).astype(float).round(4).to_string())
print("\n== SGP calibration: K/N | independence ==")
print(pd.DataFrame(r["sgp_calibration_joint"]).round(3).to_string(index=False))
print(pd.DataFrame(r["sgp_calibration_indep"]).round(3).to_string(index=False))
print("\n== SGP by season ==")
for s, v in r["sgp_by_season"].items():
    print(s, {k: round(v[k]["log_loss"], 4) for k in v}, "n", v["joint"]["n"])
print(f"\nprops void rate {r['props_void_rate']:.3f}")
for (m, q), rr in reps.items():
    if "real_lines" in rr:
        rl = rr["real_lines"]
        c = rl["calibration_two_sided"]
        print(f"\n== Real saved prices, {m} / exits {q}: {rl['n_games']} games, {rl['n_rows_priced']} priced rows ==")
        print(f"calibration (over side, decided, n={c['n']}, games={c['n_games']}): Brier model "
              f"{c['model']['brier']:.4f} vs market {c['market_devigged']['brier']:.4f}; log loss model "
              f"{c['model']['log_loss']:.4f} vs market {c['market_devigged']['log_loss']:.4f}")
        print(pd.DataFrame(c["by_market"]).T.round(4).to_string())
        for k, v in rl["roi"].items():
            print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "by_market"})
