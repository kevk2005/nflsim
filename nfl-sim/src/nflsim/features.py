"""Pregame feature construction. Everything here is strictly as-of.

Outputs (data/processed/):
  drives.parquet         one row per offensive drive (training data for the drive model)
  team_games.parquet     per team-game offensive/defensive EPA + success totals
  qb_games.parquet       per QB-game dropbacks / EPA
  games.parquet          schedule rows: identifiers, pregame-known context, and
                         clearly separated postgame labels (prefixed `label_`)
  game_features.parquet  one row per game: home_*/away_* pregame features with
                         `asof_date` = date of the latest game whose data was used

Rule: a feature for a game played on date D only uses games with date < D.
Weather (schedules temp/wind) is *actual game-time conditions*, so it is never
used as a feature. Schedules' starting-QB columns are postgame facts and are only
used to measure how often our pregame QB guess was right.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl

from .ingest import PROCESSED, load

DRIVE_RESULT_MAP = {
    "Touchdown": "TD",
    "Field goal": "FG",
    "Missed field goal": "MISSED_FG",
    "Punt": "PUNT",
    "Turnover": "TURNOVER",
    "Turnover on downs": "DOWNS",
    "End of half": "END_HALF",
    "Safety": "SAFETY",
    "Opp touchdown": "OPP_TD",
}

PBP_COLS = [
    "game_id", "season", "season_type", "week", "game_date", "home_team", "away_team",
    "posteam", "defteam", "fixed_drive", "fixed_drive_result", "drive_play_count",
    "drive_time_of_possession", "qtr", "down", "game_seconds_remaining",
    "half_seconds_remaining", "yardline_100", "score_differential", "posteam_score",
    "defteam_score", "play_type", "pass", "rush", "epa", "success", "qb_dropback",
    "qb_epa", "id", "qb_kneel", "qb_spike", "extra_point_attempt", "extra_point_result",
    "two_point_attempt", "two_point_conv_result",
]


# --------------------------------------------------------------------------- #
# Raw -> processed tables
# --------------------------------------------------------------------------- #
def load_pbp(seasons=None) -> pd.DataFrame:
    df = load("pbp", seasons)
    return df.select([c for c in PBP_COLS if c in df.columns]).to_pandas()


def load_games(seasons=None) -> pd.DataFrame:
    s = load("schedules", seasons).to_pandas()
    s["gameday"] = pd.to_datetime(s["gameday"])
    s["neutral"] = (s["location"] == "Neutral").astype(int)
    out = s[["game_id", "season", "game_type", "week", "gameday", "gametime", "home_team",
             "away_team", "neutral", "home_rest", "away_rest", "div_game", "roof"]].copy()
    # Market lines: closing numbers, used only as a comparison benchmark.
    for c in ["spread_line", "total_line", "home_moneyline", "away_moneyline"]:
        out["mkt_" + c] = s[c]
    # Postgame labels. Never used as features.
    out["label_home_score"] = s["home_score"]
    out["label_away_score"] = s["away_score"]
    out["label_home_qb_id"] = s["home_qb_id"]
    out["label_away_qb_id"] = s["away_qb_id"]
    out["label_overtime"] = s["overtime"]
    return out.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def _mmss_to_sec(x) -> float:
    if not isinstance(x, str) or ":" not in x:
        return np.nan
    m, s = x.split(":")
    return int(m) * 60 + int(s)


def build_drives(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per offensive drive with its starting state and outcome."""
    p = pbp[pbp["posteam"].notna() & pbp["fixed_drive"].notna()].copy()
    p["is_scrim"] = p["down"].notna() | p["play_type"].isin(["pass", "run", "punt", "field_goal"])
    p["is_pr"] = p["play_type"].isin(["pass", "run"]) & (p["qb_kneel"] != 1) & (p["qb_spike"] != 1)
    key = ["game_id", "fixed_drive"]
    # First scrimmage play gives the starting state; fall back to the first play.
    first_scrim = p[p["is_scrim"]].groupby(key, sort=False).head(1).set_index(key)
    first_any = p.groupby(key, sort=False).head(1).set_index(key)
    start = first_scrim.reindex(first_any.index)
    for c in ["yardline_100", "game_seconds_remaining", "half_seconds_remaining", "qtr",
              "score_differential", "posteam", "defteam"]:
        start[c] = start[c].fillna(first_any[c]) if c in start else first_any[c]
    agg = p.groupby(key, sort=False).agg(n_pr=("is_pr", "sum"))
    pr = p[p["is_pr"]]
    agg["n_pass_pr"] = pr.groupby(key)["pass"].sum().reindex(agg.index).fillna(0)
    d = first_any[["season", "season_type", "week", "game_date", "home_team", "away_team",
                   "fixed_drive_result", "drive_play_count", "drive_time_of_possession"]].copy()
    for c in ["yardline_100", "game_seconds_remaining", "half_seconds_remaining", "qtr",
              "score_differential", "posteam", "defteam"]:
        d[c] = start[c]
    d["n_plays_pr"] = agg["n_pr"]
    d["n_pass_pr"] = agg["n_pass_pr"]
    d = d.reset_index()
    d["result"] = d["fixed_drive_result"].map(DRIVE_RESULT_MAP)
    d["duration"] = d["drive_time_of_possession"].map(_mmss_to_sec)
    d["plays"] = d["drive_play_count"]
    d["is_home"] = (d["posteam"] == d["home_team"]).astype(int)
    d = d.dropna(subset=["result", "yardline_100", "half_seconds_remaining"])
    d = d.sort_values(["game_id", "fixed_drive"]).reset_index(drop=True)
    d["drive_uid"] = np.arange(len(d))
    # Next drive's start (same game, same half) -> field-position transition table.
    d["half"] = np.where(d["qtr"] <= 2, 1, np.where(d["qtr"] <= 4, 2, 3))
    nxt = d.groupby("game_id")[["yardline_100", "half"]].shift(-1)
    d["next_start_yl"] = np.where(nxt["half"] == d["half"], nxt["yardline_100"], np.nan)
    d["game_date"] = pd.to_datetime(d["game_date"])
    keep = ["drive_uid", "game_id", "season", "season_type", "week", "game_date", "fixed_drive", "posteam",
            "defteam", "is_home", "qtr", "half", "game_seconds_remaining",
            "half_seconds_remaining", "yardline_100", "score_differential", "result",
            "plays", "duration", "n_plays_pr", "n_pass_pr", "next_start_yl"]
    return d[keep]


def build_pat_table(pbp: pd.DataFrame) -> pd.DataFrame:
    """PAT / 2-pt attempts with score state (score_differential is pre-attempt, +6 applied)."""
    x = pbp[(pbp["extra_point_attempt"] == 1) | (pbp["two_point_attempt"] == 1)].copy()
    x["go2"] = (x["two_point_attempt"] == 1).astype(int)
    x["good"] = np.where(x["go2"] == 1, x["two_point_conv_result"] == "success",
                         x["extra_point_result"] == "good").astype(int)
    x["game_date"] = pd.to_datetime(x["game_date"])
    return x[["game_id", "season", "game_date", "qtr", "game_seconds_remaining",
              "score_differential", "go2", "good"]]


def build_team_games(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per team-game offensive and defensive EPA/success over pass+run plays."""
    p = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna()
            & (pbp["qb_kneel"] != 1) & (pbp["qb_spike"] != 1)]
    g = p.groupby(["game_id", "posteam", "defteam"]).agg(
        plays=("epa", "size"), epa=("epa", "sum"), succ=("success", "sum")).reset_index()
    return g


def build_qb_games(pbp: pd.DataFrame) -> pd.DataFrame:
    p = pbp[(pbp["qb_dropback"] == 1) & pbp["id"].notna() & pbp["qb_epa"].notna()]
    g = p.groupby(["game_id", "posteam", "id"]).agg(
        dropbacks=("qb_epa", "size"), qb_epa=("qb_epa", "sum")).reset_index()
    return g.rename(columns={"id": "qb_id", "posteam": "team"})


# --------------------------------------------------------------------------- #
# Sequential as-of team ratings
# --------------------------------------------------------------------------- #
@dataclass
class RatingParams:
    halflife_games: float = 8.0     # exponential recency weighting, in team games
    season_carry: float = 0.5       # extra weight multiplier per season boundary
    prior_plays: float = 400.0      # shrinkage strength toward league average, in plays
    qb_halflife_games: float = 12.0
    qb_prior_dropbacks: float = 250.0
    qb_prior_offset: float = -0.10  # unknown/new QBs are shrunk below league average


def _weighted(obs: list, cur_season: int, p: RatingParams, n_idx: int, val_idx: list[int],
              halflife: float, prior_n: float, prior_vals: list[float]) -> list[float]:
    """Shrunk, recency-weighted per-unit averages from a list of observation tuples.

    obs entries: (season, n, v1_sum, v2_sum, ...). Most recent last.
    """
    if not obs:
        return list(prior_vals)
    k = len(obs)
    arr = np.asarray(obs, dtype=float)
    games_ago = np.arange(k - 1, -1, -1)
    w = 0.5 ** (games_ago / halflife) * p.season_carry ** (cur_season - arr[:, 0])
    n = arr[:, n_idx]
    wn = (w * n).sum()
    out = []
    for vi, pv in zip(val_idx, prior_vals):
        out.append(((w * arr[:, vi]).sum() + prior_n * pv) / (wn + prior_n))
    return out


def compute_team_ratings(games: pd.DataFrame, team_games: pd.DataFrame,
                         p: RatingParams | None = None, max_obs: int = 60) -> pd.DataFrame:
    """For every game and both teams, ratings using only games strictly before it.

    Opponent adjustment: each past game's offensive EPA/play is adjusted by the
    opponent's *pregame* defensive rating at that time (and vice versa), relative
    to league average. Ratings are then a recency-weighted average of those
    adjusted values, shrunk toward the league average with `prior_plays`.
    """
    p = p or RatingParams()
    tg = team_games.set_index(["game_id", "posteam"])
    obs_off = defaultdict(list)   # team -> [(season, plays, adj_epa_sum, adj_succ_sum)]
    obs_def = defaultdict(list)
    last_date = {}
    league_hist = []              # (date, plays, epa_sum, succ_sum) for league averages
    rows = []

    for day, day_games in games.groupby("gameday", sort=True):
        # League average from the trailing ~1 year of plays, strictly before today.
        lo = day - pd.Timedelta(days=380)
        recent = [x for x in league_hist if lo <= x[0] < day]
        if recent:
            n = sum(x[1] for x in recent)
            lg_epa = sum(x[2] for x in recent) / n
            lg_sr = sum(x[3] for x in recent) / n
        else:
            lg_epa, lg_sr = 0.0, 0.43
        pre = {}
        for _, g in day_games.iterrows():
            for side, team in (("home", g.home_team), ("away", g.away_team)):
                off = _weighted(obs_off[team][-max_obs:], g.season, p, 1, [2, 3],
                                p.halflife_games, p.prior_plays, [lg_epa, lg_sr])
                de = _weighted(obs_def[team][-max_obs:], g.season, p, 1, [2, 3],
                               p.halflife_games, p.prior_plays, [lg_epa, lg_sr])
                pre[(g.game_id, team)] = (off, de)
                rows.append({
                    "game_id": g.game_id, "team": team, "side": side,
                    "off_epa": off[0], "off_sr": off[1], "def_epa": de[0], "def_sr": de[1],
                    "lg_epa": lg_epa, "lg_sr": lg_sr,
                    "n_obs": len(obs_off[team]),
                    "asof_date": last_date.get(team, pd.NaT),
                })
        # After all of today's pregame features are recorded, ingest today's results.
        for _, g in day_games.iterrows():
            if pd.isna(g.label_home_score):
                continue
            for team, opp in ((g.home_team, g.away_team), (g.away_team, g.home_team)):
                if (g.game_id, team) not in tg.index:
                    continue
                r = tg.loc[(g.game_id, team)]
                opp_off, opp_def = pre[(g.game_id, opp)]
                n = r["plays"]
                adj_off_epa = r["epa"] - n * (opp_def[0] - lg_epa)
                adj_off_sr = r["succ"] - n * (opp_def[1] - lg_sr)
                obs_off[team].append((g.season, n, adj_off_epa, adj_off_sr))
                # Defense: plays faced by `opp` offense against `team`.
                if (g.game_id, opp) in tg.index:
                    ro = tg.loc[(g.game_id, opp)]
                    m = ro["plays"]
                    adj_def_epa = ro["epa"] - m * (opp_off[0] - lg_epa)
                    adj_def_sr = ro["succ"] - m * (opp_off[1] - lg_sr)
                    obs_def[team].append((g.season, m, adj_def_epa, adj_def_sr))
                last_date[team] = day
                league_hist.append((day, n, r["epa"], r["succ"]))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Starting QB (pregame estimate) and QB ratings
# --------------------------------------------------------------------------- #
def _depth_chart_qbs(games: pd.DataFrame) -> dict:
    """(game_id, team) -> ordered list of depth-chart QBs known before the game."""
    out = {}
    seasons = sorted(games["season"].unique())
    old = [s for s in seasons if s <= 2024]
    new = [s for s in seasons if s >= 2025]
    if old:
        dc = load("depth_charts", old).to_pandas()
        dc = dc[(dc["position"] == "QB") & (dc["formation"] == "Offense") & dc["week"].notna()]
        dc["rank"] = pd.to_numeric(dc["depth_team"], errors="coerce")
        dc = dc.sort_values("rank")
        lookup = dc.groupby(["season", "week", "club_code"])["gsis_id"].apply(list).to_dict()
        for g in games[games["season"] <= 2024].itertuples():
            for t in (g.home_team, g.away_team):
                lst = lookup.get((g.season, g.week, t))
                if lst:
                    out[(g.game_id, t)] = list(dict.fromkeys(x for x in lst if isinstance(x, str)))
    if new:
        try:
            dc = load("depth_charts", new).to_pandas()
        except FileNotFoundError:
            dc = None
        if dc is not None:
            dc = dc[dc["pos_abb"] == "QB"].copy()
            dc["ts"] = pd.to_datetime(dc["dt"], utc=True).dt.tz_localize(None)
            dc = dc.sort_values(["ts", "pos_rank"])
            by_team = {t: x for t, x in dc.groupby("team")}
            for g in games[games["season"] >= 2025].itertuples():
                for t in (g.home_team, g.away_team):
                    x = by_team.get(t)
                    if x is None:
                        continue
                    # Snapshots taken before the game day (conservative: before 00:00 that day).
                    x = x[x["ts"] < g.gameday]
                    if x.empty:
                        continue
                    snap = x[x["ts"] == x["ts"].max()].sort_values("pos_rank")
                    ids = [i for i in snap["gsis_id"] if isinstance(i, str)]
                    if ids:
                        out[(g.game_id, t)] = list(dict.fromkeys(ids))
    return out


def _injury_out(games: pd.DataFrame) -> set:
    """(season, week, team, gsis_id) for players whose final weekly status is Out.

    Assumption (documented): the final weekly injury report status is known before kickoff.
    """
    try:
        inj = load("injuries", sorted(games["season"].unique())).to_pandas()
    except FileNotFoundError:
        return set()
    inj = inj[inj["report_status"] == "Out"]
    return set(zip(inj["season"], inj["week"], inj["team"], inj["gsis_id"]))


def compute_qb_features(games: pd.DataFrame, qb_games: pd.DataFrame,
                        team_ratings: pd.DataFrame, p: RatingParams | None = None) -> pd.DataFrame:
    p = p or RatingParams()
    dc = _depth_chart_qbs(games)
    out_set = _injury_out(games)
    qbg = qb_games.merge(games[["game_id", "season", "gameday"]], on="game_id")
    primary = (qbg.sort_values("dropbacks", ascending=False)
               .groupby(["game_id", "team"]).head(1).set_index(["game_id", "team"])["qb_id"])
    qb_by_game = {k: v for k, v in qbg.groupby("game_id")}
    lg = team_ratings.drop_duplicates("game_id").set_index("game_id")["lg_epa"]

    obs = defaultdict(list)          # qb -> [(season, dropbacks, epa_sum)]
    last_primary = {}                # team -> qb id of primary passer in its previous game
    team_hist = defaultdict(list)    # team -> [(season, {qb_id: dropbacks})], one per game
    rows = []
    for day, day_games in games.groupby("gameday", sort=True):
        for g in day_games.itertuples():
            prior_mean = lg.get(g.game_id, 0.0) + p.qb_prior_offset

            def rating(q):
                if q is None:
                    return prior_mean
                return _weighted(obs[q][-40:], g.season, p, 1, [2], p.qb_halflife_games,
                                 p.qb_prior_dropbacks, [prior_mean])[0]

            for t in (g.home_team, g.away_team):
                cand = dc.get((g.game_id, t), [])
                src = "depth_chart"
                qb = None
                qb2 = None   # backup: next healthy QB on the pregame depth chart
                for i, q in enumerate(cand):
                    if (g.season, g.week, t, q) in out_set:
                        if qb is None:
                            src = "depth_chart+injury"
                        continue
                    if qb is None:
                        qb = q
                    elif q != qb:
                        qb2 = q
                        break
                if qb is None:
                    qb = last_primary.get(t)
                    src = "prev_game" if qb else "none"
                r_qb = rating(qb)
                # qb_delta compares the expected starter with the dropback-weighted mix
                # of QBs in the games behind the team's offense rating (same recency
                # and season weights, same 60-game window). Comparing only with last
                # game's QB missed backups on their 2nd+ start, whose team rating
                # still mostly reflects the injured starter.
                mix = defaultdict(float)
                hist = team_hist[t][-60:]
                for k, (s_, qbs) in enumerate(hist):
                    w = 0.5 ** ((len(hist) - 1 - k) / p.halflife_games) * \
                        p.season_carry ** (g.season - s_)
                    for q, db in qbs.items():
                        mix[q] += w * db
                wsum = sum(mix.values())
                if wsum > 0:
                    r_mix = sum(w * rating(q) for q, w in mix.items()) / wsum
                    share = mix.get(qb, 0.0) / wsum
                else:
                    r_mix, share = r_qb, np.nan
                rows.append({"game_id": g.game_id, "team": t, "qb_id": qb, "qb_source": src,
                             "qb_epa": r_qb, "qb_delta": r_qb - r_mix,
                             "qb_n_obs": len(obs[qb]) if qb else 0,
                             "qb_changed": qb != last_primary.get(t),
                             "qb_mix_share": share,
                             # backup rating; unknown backup -> the unknown-QB prior
                             "qb2_id": qb2, "qb2_epa": rating(qb2)})
        for g in day_games.itertuples():
            gq = qb_by_game.get(g.game_id)
            if gq is None:
                continue
            for r in gq.itertuples():
                obs[r.qb_id].append((g.season, r.dropbacks, r.qb_epa))
            for t, tq in gq.groupby("team"):
                team_hist[t].append((g.season, dict(zip(tq["qb_id"], tq["dropbacks"]))))
            for t in (g.home_team, g.away_team):
                if (g.game_id, t) in primary.index:
                    last_primary[t] = primary.loc[(g.game_id, t)]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Assemble
# --------------------------------------------------------------------------- #
def assemble_game_features(games: pd.DataFrame, ratings: pd.DataFrame,
                           qbf: pd.DataFrame) -> pd.DataFrame:
    r = ratings.merge(qbf, on=["game_id", "team"], how="left")
    cols = ["team", "off_epa", "off_sr", "def_epa", "def_sr", "n_obs", "asof_date",
            "qb_id", "qb_source", "qb_epa", "qb_delta", "qb_n_obs", "qb_changed",
            "qb_mix_share", "qb2_id", "qb2_epa"]
    h = r[r["side"] == "home"][["game_id", "lg_epa", "lg_sr"] + cols]
    a = r[r["side"] == "away"][["game_id"] + cols]
    h = h.rename(columns={c: "home_" + c for c in cols})
    a = a.rename(columns={c: "away_" + c for c in cols})
    base = games[["game_id", "season", "game_type", "week", "gameday", "home_team",
                  "away_team", "neutral", "home_rest", "away_rest", "div_game"]]
    f = base.merge(h.drop(columns=["home_team"]), on="game_id").merge(
        a.drop(columns=["away_team"]), on="game_id")
    f["asof_date"] = f[["home_asof_date", "away_asof_date"]].max(axis=1)
    f["weather"] = "unavailable"  # forecasts not in nflverse; actuals never used
    return f


def build_all(seasons=None, params: RatingParams | None = None, verbose=True) -> dict:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    pbp = load_pbp(seasons)
    games = load_games(seasons)
    drives = build_drives(pbp)
    pat = build_pat_table(pbp)
    tg = build_team_games(pbp)
    qbg = build_qb_games(pbp)
    if verbose:
        print(f"drives={len(drives)} team_games={len(tg)} qb_games={len(qbg)}")
    ratings = compute_team_ratings(games, tg, params)
    qbf = compute_qb_features(games, qbg, ratings, params)
    feats = assemble_game_features(games, ratings, qbf)
    out = {"drives": drives, "pat": pat, "team_games": tg, "qb_games": qbg,
           "games": games, "game_features": feats}
    for k, v in out.items():
        v.to_parquet(PROCESSED / f"{k}.parquet", index=False)
    return out


def read_processed(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"{name}.parquet")


if __name__ == "__main__":
    build_all()
