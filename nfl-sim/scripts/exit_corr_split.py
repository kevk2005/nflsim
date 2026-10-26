"""Diagnostic (no model changes): QB ~ receiver / total correlations split by
whether the starting QB left the game early, actual vs simulated, 2024-25.

Actual exit = permanent exit of the team's starting QB (same definition as
scripts/qb_exits.py). Simulated exit = the simulated starter left in that
simulation. Uses the saved independent player walk-forward
(data/processed/player_wf_gamecorr.pkl), so run the --players backtest first.
"""
import numpy as np
import pandas as pd

from nflsim.ingest import PROCESSED, load

p = load("pbp", [2024, 2025]).select(["game_id", "play_id", "posteam", "qb_dropback", "qb_kneel",
                                      "passer_player_id", "rusher_player_id"]).to_pandas()
q = p[((p.qb_dropback == 1) | (p.qb_kneel == 1)) & p.posteam.notna()].sort_values(["game_id", "play_id"])
q["qb"] = q.passer_player_id.fillna(q.rusher_player_id)
q = q.dropna(subset=["qb"])
exit_actual = {}
for (gid, team), x in q.groupby(["game_id", "posteam"], sort=False):
    ids = x.qb.to_numpy()
    last = np.flatnonzero(ids == ids[0]).max()
    exit_actual[(gid, team)] = bool((ids[last + 1:] != ids[0]).any())

gc = pd.read_pickle(PROCESSED / "player_wf_gamecorr.pkl")
pairs = [("QB1_pass_yards", "WR1_rec_yards"), ("QB1_pass_yards", "WR2_rec_yards"),
         ("QB1_pass_yards", "TE1_rec_yards"), ("QB1_pass_yards", "total")]
rows = []
for a, b in pairs:
    acc = {k: ([], [], [], []) for k in ("no_exit", "exit")}   # act_a, act_b, sim_a, sim_b
    for rec in gc:
        def val(k, act):
            if k == "total":
                return rec["act_total" if act else "sim_total"]
            return rec.get(("act_" if act else "") + k)
        xa, xb = val(a, True), val(b, True)
        if xa is None or xb is None or np.isnan(xa) or np.isnan(xb):
            continue
        ex = exit_actual.get((rec["game_id"], rec["team"]))
        if ex is None:
            continue
        acc["exit" if ex else "no_exit"][0].append(xa)
        acc["exit" if ex else "no_exit"][1].append(xb)
        sa, sb, se = np.asarray(val(a, False), float), np.asarray(val(b, False), float), rec["sim_qb_exit"]
        ok = ~np.isnan(sa) & ~np.isnan(sb)
        for k, m in (("exit", se & ok), ("no_exit", ~se & ok)):
            acc[k][2].extend(sa[m]); acc[k][3].extend(sb[m])
    for k in ("no_exit", "exit", "all"):
        if k == "all":
            A = [acc["no_exit"][i] + acc["exit"][i] for i in range(4)]
        else:
            A = acc[k]
        rows.append({"pair": f"{a} ~ {b}", "games": k,
                     "n_actual_team_games": len(A[0]), "actual_corr": np.corrcoef(A[0], A[1])[0, 1],
                     "n_sim_samples": len(A[2]), "sim_corr": np.corrcoef(A[2], A[3])[0, 1]})
r = pd.DataFrame(rows)
r["gap_sim_minus_actual"] = r.sim_corr - r.actual_corr
pd.set_option("display.width", 200)
print(r.round(3).to_string(index=False))
exit_share = r[r.games == "exit"].n_actual_team_games.iloc[0] / r[r.games == "all"].n_actual_team_games.iloc[0]
sim_share = r[r.games == "exit"].n_sim_samples.iloc[0] / r[r.games == "all"].n_sim_samples.iloc[0]
print(f"share of team-games with an exit: actual {exit_share:.3f}, simulated {sim_share:.3f}")
r.to_csv(PROCESSED / "exit_corr_split.csv", index=False)
