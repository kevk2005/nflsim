"""How often does the starting QB leave before the game ends, and when?

Estimated on 2018-2020 (before every test window). The starter is the QB of the
team's first dropback/kneel. A permanent exit = another QB takes a dropback or
kneel after the starter's last one. The exit drive is the drive of that first
play by the replacement. Per-drive exit hazards are computed by the state at
the start of each drive the starter began on the field.
Output: tables printed + JSON written to data/processed/qb_exit_estimates_<first>_<last>.json.
"""
import json
import sys

import numpy as np
import pandas as pd
import polars as pl

from nflsim.ingest import PROCESSED, load

seasons = [int(x) for x in sys.argv[1:]] or [2018, 2019, 2020]
p = load("pbp", seasons).select([
    "game_id", "play_id", "fixed_drive", "posteam", "qb_dropback", "qb_kneel", "passer_player_id",
    "rusher_player_id", "game_seconds_remaining", "score_differential", "qtr"]).to_pandas()
q = p[((p.qb_dropback == 1) | (p.qb_kneel == 1)) & p.posteam.notna()].copy()
q["qb"] = q.passer_player_id.fillna(q.rusher_player_id)
q = q.dropna(subset=["qb"]).sort_values(["game_id", "play_id"])
drives = pd.read_parquet(PROCESSED / "drives.parquet")
drives = drives[drives.season.isin(seasons) & (drives.half <= 2)]

exits, team_games = [], 0
exit_drive = {}
for (gid, team), x in q.groupby(["game_id", "posteam"], sort=False):
    team_games += 1
    starter = x.qb.iloc[0]
    last_s = np.flatnonzero(x.qb.to_numpy() == starter).max()
    after = x.iloc[last_s + 1:]
    others_before = (x.iloc[:last_s].qb != starter).any()
    if len(after):
        r = after.iloc[0]
        exits.append({"game_id": gid, "team": team, "secs_left": r.game_seconds_remaining,
                      "diff": r.score_differential, "qtr": r.qtr, "drive": r.fixed_drive,
                      "temp_absence_before": bool(others_before)})
        exit_drive[(gid, team)] = r.fixed_drive
ex = pd.DataFrame(exits)
print(f"seasons {seasons}: team-games {team_games}, permanent starter exits {len(ex)} "
      f"({len(ex) / team_games:.1%})")

def blowout(diff, secs):
    return (np.abs(diff) >= 17) & (secs <= 900) | (np.abs(diff) >= 24) & (secs <= 1800)

ex["blowout"] = blowout(ex["diff"], ex["secs_left"])
ex["kind"] = np.where(ex.blowout, "blowout", "injury/benching")
print(ex.groupby("kind").size().rename("n").to_frame().assign(share_of_team_games=lambda d: d.n / team_games))
ex["when"] = pd.cut(ex.secs_left, [-1, 120, 300, 900, 1800, 2700, 3600],
                    labels=["last 2:00", "2-5 min left", "rest of Q4", "Q3", "Q2", "Q1"])
print(pd.crosstab(ex["when"], ex["kind"]))
print("score diff (exiting team's view) at exit, blowout exits: leading", int((ex.blowout & (ex["diff"] > 0)).sum()),
      "trailing", int((ex.blowout & (ex["diff"] < 0)).sum()))

# Per-drive hazards by drive-start state, for drives the starter started on the field.
d = drives.copy()
d["exit_drive"] = d.set_index(["game_id", "posteam"]).index.map(exit_drive)
d = d[d["exit_drive"].isna() | (d.fixed_drive <= d["exit_drive"])]       # at risk
d["exit_here"] = d.fixed_drive == d["exit_drive"]
secs = np.where(d.half == 1, 1800 + d.half_seconds_remaining, d.half_seconds_remaining)
diff = d.score_differential.to_numpy()
ad = np.abs(diff)
bucket = np.select(
    [(ad >= 24) & (secs <= 1800) & (secs > 900), (ad >= 17) & (secs <= 900) & (secs > 300),
     (ad >= 17) & (secs <= 300), (ad >= 24) & (secs <= 900)],
    ["blow24_Q3", "blow17_Q4", "blow17_last5", "blow24_Q4"], "normal")
# order matters: 24+ in Q4 overrides 17+
bucket = np.where((ad >= 24) & (secs <= 900) & (secs > 300), "blow24_Q4", bucket)
bucket = np.where((ad >= 24) & (secs <= 300), "blow24_last5", bucket)
d["bucket"] = bucket
h = d.groupby("bucket").agg(drives=("exit_here", "size"), exits=("exit_here", "sum"))
h["hazard_per_drive"] = h.exits / h.drives
print(h)
# Refinement: blowout buckets split by leading / trailing; normal states by quarter.
d["side"] = np.where(diff > 0, "leading", "trailing")
d["qtr_"] = np.select([secs > 2700, secs > 1800, secs > 900], ["Q1", "Q2", "Q3"], "Q4")
hb = d[d.bucket != "normal"].groupby(["bucket", "side"]).agg(drives=("exit_here", "size"), exits=("exit_here", "sum"))
hb["hazard_per_drive"] = hb.exits / hb.drives
hn = d[d.bucket == "normal"].groupby("qtr_").agg(drives=("exit_here", "size"), exits=("exit_here", "sum"))
hn["hazard_per_drive"] = hn.exits / hn.drives
print(hb); print(hn)
# injury/benching exits in "normal" states: when in the game (for reporting)
out = {"seasons": seasons, "team_games": team_games, "permanent_exits": int(len(ex)),
       "by_kind": ex.groupby("kind").size().to_dict(),
       "hazard_per_drive": {k: float(v) for k, v in h.hazard_per_drive.items()},
       "drives_at_risk": {k: int(v) for k, v in h.drives.items()},
       "exits_in_bucket": {k: int(v) for k, v in h.exits.items()},
       "blowout_by_side": {f"{b}|{sd}": [int(r.exits), int(r.drives), float(r.hazard_per_drive)]
                           for (b, sd), r in hb.iterrows()},
       "normal_by_quarter": {q_: [int(r.exits), int(r.drives), float(r.hazard_per_drive)]
                             for q_, r in hn.iterrows()}}
(PROCESSED / f"qb_exit_estimates_{seasons[0]}_{seasons[-1]}.json").write_text(json.dumps(out, indent=2))
