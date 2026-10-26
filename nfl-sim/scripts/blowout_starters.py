"""After the starting QB leaves, do the other starters keep their usage?
2018-2020. For each team-game with a permanent starter exit, compare each
pregame-role player's share of team targets / carries before vs after the exit,
separately for blowout and injury/benching exits."""
import numpy as np
import pandas as pd

from nflsim.ingest import PROCESSED, load

seasons = [2018, 2019, 2020]
p = load("pbp", seasons).select(["game_id", "play_id", "posteam", "qb_dropback", "qb_kneel", "passer_player_id",
                                 "rusher_player_id", "receiver_player_id", "play_type", "qb_scramble",
                                 "game_seconds_remaining", "score_differential"]).to_pandas()
p = p[p.posteam.notna()].sort_values(["game_id", "play_id"])
roles = pd.read_parquet(PROCESSED / "roles.parquet")
roles = roles[roles.position.isin(["WR", "TE", "RB"])]
roles["role"] = roles.position + np.minimum(roles["rank"], 3).astype(str)
rows = []
for (gid, team), x in p.groupby(["game_id", "posteam"], sort=False):
    qb = x[(x.qb_dropback == 1) | (x.qb_kneel == 1)]
    qid = qb.passer_player_id.fillna(qb.rusher_player_id)
    if qid.isna().all():
        continue
    starter = qid.dropna().iloc[0]
    last = qb.play_id[qid == starter].max()
    after_q = qb[(qb.play_id > last) & (qid != starter)]
    if after_q.empty:
        continue
    t0 = after_q.play_id.iloc[0]
    r0 = after_q.iloc[0]
    blow = (abs(r0.score_differential) >= 17 and r0.game_seconds_remaining <= 900) or \
           (abs(r0.score_differential) >= 24 and r0.game_seconds_remaining <= 1800)
    tg = x[x.receiver_player_id.notna() & (x.play_type == "pass")]
    ru = x[(x.play_type == "run") & (x.qb_scramble != 1) & x.rusher_player_id.notna()]
    rl = roles[(roles.game_id == gid) & (roles.team == team)]
    for per, sl in (("before", lambda d: d[d.play_id < t0]), ("after", lambda d: d[d.play_id >= t0])):
        a, b = sl(tg), sl(ru)
        for _, r in rl.iterrows():
            rows.append({"game_id": gid, "team": team, "kind": "blowout" if blow else "injury/benching",
                         "role": r.role, "period": per,
                         "tgt": (a.receiver_player_id == r.player_id).sum(), "team_tgt": len(a),
                         "car": (b.rusher_player_id == r.player_id).sum(), "team_car": len(b)})
d = pd.DataFrame(rows)
g = d.groupby(["kind", "role", "period"])[["tgt", "team_tgt", "car", "team_car"]].sum()
g["tgt_share"] = g.tgt / g.team_tgt.replace(0, np.nan)
g["car_share"] = g.car / g.team_car.replace(0, np.nan)
w = g[["tgt_share", "car_share"]].unstack("period")
out = pd.DataFrame({
    "tgt_before": w[("tgt_share", "before")], "tgt_after": w[("tgt_share", "after")],
    "car_before": w[("car_share", "before")], "car_after": w[("car_share", "after")]})
out["tgt_ratio"] = out.tgt_after / out.tgt_before
out["car_ratio"] = out.car_after / out.car_before
n = d.drop_duplicates(["game_id", "team", "kind"]).groupby("kind").size()
print("team-games:", n.to_dict())
team_after = d[d.period == "after"].drop_duplicates(["game_id", "team"]).groupby("kind")[["team_tgt", "team_car"]].sum()
print("team targets / carries after exit:", team_after.to_dict())
print(out.round(3).to_string())
out.to_csv(PROCESSED / "blowout_starter_ratios.csv")
