"""Player layer: turns simulated drives into player box scores.

How a simulated drive becomes plays
-----------------------------------
The simulator samples each drive from a real historical drive with the same
outcome, game state and starting field-position bucket (PaceTable). That
drive's play-by-play is the *template*: the sequence of play types (pass with a
target, throwaway/spike, sack, designed run, scramble, kneel), completions,
interceptions, yards and the touchdown play. Template yards are rescaled so a
touchdown drive gains exactly the simulated distance to the goal line, and a
non-scoring drive never gains more than that distance.

Each template play is then assigned to a player of the simulated offense:
  dropbacks / scrambles / kneels -> the expected starting QB
  targeted passes -> receiver r with P(r | complete?, yards bin, TD?) proportional to
                     target_share_r * P(complete | r) * P(yards bin | r) * TD ratio_r
  designed runs   -> carrier r with P(r | yards bin, TD?) proportional to
                     carry_share_r * P(yards bin | r) * TD ratio_r
Shares and efficiency come from each player's recent games (recency-weighted),
shrunk toward the average for his pregame role (position + depth-chart rank).
Players listed Out or Doubtful on the final injury report get zero, and their
share goes to teammates through re-normalization.

Because every player stat is built from the same plays, consistency holds by
construction: receptions <= targets, team completions = sum of receptions,
QB passing yards = sum of receiving yards, passing TDs = receiving TDs, and
offensive TD plays match the drive's touchdown.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .ingest import PROCESSED, load

# Play kinds in a template
PASS, PASS_NT, SACK, RUN, SCRAM, KNEEL = range(6)
SKILL = ["QB", "RB", "FB", "WR", "TE"]
REC_BINS = np.array([-np.inf, 0.5, 4.5, 9.5, 14.5, 19.5, 29.5, np.inf])     # 7 bins
RUSH_BINS = np.array([-np.inf, -0.5, 2.5, 5.5, 9.5, 19.5, np.inf])         # 6 bins
NRB, NUB = len(REC_BINS) - 1, len(RUSH_BINS) - 1
ROLE_CAP = {"WR": 5, "TE": 3, "RB": 3, "FB": 1, "QB": 2}
UNLISTED = 9   # rank for active players not on the depth chart


@dataclass
class PlayerParams:
    halflife_games: float = 6.0
    season_carry: float = 0.5
    max_games: int = 24
    k_share: float = 60.0      # prior strength in team targets/carries
    k_catch: float = 40.0      # in targets
    k_bins: float = 30.0       # in receptions / carries
    k_td: float = 30.0         # in receptions / carries
    k_active: float = 2.0      # prior strength (in games) for P(active)
    active_window: int = 4     # team games this season used for P(active)
    # Game-to-game share variation: per-simulation shares ~ Dirichlet(alpha * share).
    # Estimated on 2023 by method of moments (scripts/estimate_dispersion.py); None = off.
    alpha_tgt: float | None = 63.5
    alpha_car: float | None = 11.35
# k_* and the recency settings were set by hand, not tuned (see README).


# --------------------------------------------------------------------------- #
# Building tables
# --------------------------------------------------------------------------- #
def build_play_tables(pbp_raw, drives: pd.DataFrame):
    """drive_plays (templates) and pbp-derived player-game usage."""
    import polars as pl
    cols = ["game_id", "play_id", "fixed_drive", "posteam", "play_type", "pass", "rush",
            "qb_scramble", "sack", "complete_pass", "interception", "pass_touchdown",
            "rush_touchdown", "yards_gained", "receiver_player_id", "rusher_player_id",
            "passer_player_id", "two_point_attempt", "qb_kneel", "qb_spike"]
    p = pbp_raw.select(cols).filter(
        pl.col("play_type").is_in(["pass", "run", "qb_kneel", "qb_spike"])
        & (pl.col("two_point_attempt").fill_null(0) == 0)
        & pl.col("posteam").is_not_null()).to_pandas()
    kind = np.select(
        [p["qb_kneel"] == 1, p["qb_spike"] == 1, p["sack"] == 1, p["qb_scramble"] == 1,
         p["play_type"] == "run", p["receiver_player_id"].notna()],
        [KNEEL, PASS_NT, SACK, SCRAM, RUN, PASS], PASS_NT)
    p["kind"] = kind
    p["complete"] = ((p["complete_pass"] == 1) & (p["kind"] == PASS)).astype(np.int8)
    p["intercept"] = (p["interception"] == 1).astype(np.int8)
    p["td"] = (((p["pass_touchdown"] == 1) & (p["kind"] == PASS))
               | ((p["rush_touchdown"] == 1) & np.isin(p["kind"], [RUN, SCRAM]))).astype(np.int8)
    p["yards"] = p["yards_gained"].fillna(0).astype(float)
    p.loc[(p["kind"] == PASS) & (p["complete"] == 0), "yards"] = 0.0
    p.loc[p["kind"] == PASS_NT, "yards"] = 0.0
    key = drives[["game_id", "fixed_drive", "drive_uid", "game_date", "season"]]
    p = p.merge(key, on=["game_id", "fixed_drive"], how="inner").sort_values(["drive_uid", "play_id"])
    plays = p[["drive_uid", "kind", "yards", "complete", "intercept", "td"]].reset_index(drop=True)

    # Player-game usage (targets incl. intercepted, designed carries)
    t = p[p["kind"] == PASS]
    t = t.assign(bin=np.digitize(t["yards"], REC_BINS[1:-1]))
    # incomplete targets already carry 0 yards
    rec = t.groupby(["game_id", "posteam", "receiver_player_id"]).agg(
        targets=("kind", "size"), receptions=("complete", "sum"),
        rec_yards=("yards", "sum"), rec_td=("td", "sum")).reset_index()
    tc = t[t["complete"] == 1]
    rb = (tc.groupby(["game_id", "posteam", "receiver_player_id", "bin"]).size()
          .unstack(fill_value=0).reindex(columns=range(NRB), fill_value=0))
    rb.columns = [f"rb{i}" for i in range(NRB)]
    rec = rec.merge(rb.reset_index(), on=["game_id", "posteam", "receiver_player_id"], how="left")
    rec = rec.rename(columns={"receiver_player_id": "player_id"})
    r = p[p["kind"] == RUN]
    r = r.assign(bin=np.digitize(r["yards"], RUSH_BINS[1:-1]))
    car = r.groupby(["game_id", "posteam", "rusher_player_id"]).agg(
        carries=("kind", "size"), rush_yards=("yards", "sum"), rush_td=("td", "sum")).reset_index()
    ub = (r.groupby(["game_id", "posteam", "rusher_player_id", "bin"]).size()
          .unstack(fill_value=0).reindex(columns=range(NUB), fill_value=0))
    ub.columns = [f"ub{i}" for i in range(NUB)]
    car = car.merge(ub.reset_index(), on=["game_id", "posteam", "rusher_player_id"], how="left")
    car = car.rename(columns={"rusher_player_id": "player_id"})
    pg = rec.merge(car, on=["game_id", "posteam", "player_id"], how="outer")
    team_tot = p.groupby(["game_id", "posteam"]).agg(
        team_targets=("kind", lambda s: int((s == PASS).sum())),
        team_carries=("kind", lambda s: int((s == RUN).sum()))).reset_index()
    pg = pg.rename(columns={"posteam": "team"})
    return plays, pg, team_tot.rename(columns={"posteam": "team"})


def _pfr_map() -> dict:
    r = load("rosters").to_pandas()[["pfr_id", "gsis_id"]].dropna().drop_duplicates("pfr_id")
    return dict(zip(r["pfr_id"], r["gsis_id"]))


def _positions() -> dict:
    r = load("rosters").to_pandas()[["gsis_id", "position", "season"]].dropna()
    r = r.sort_values("season").drop_duplicates("gsis_id", keep="last")
    return dict(zip(r["gsis_id"], r["position"]))


def build_player_games(pg: pd.DataFrame, team_tot: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Usage per player-game, including active games with zero usage (from snap counts)."""
    sc = load("snap_counts").to_pandas()
    sc = sc[sc["offense_snaps"] > 0]
    pm = _pfr_map()
    sc["player_id"] = sc["pfr_player_id"].map(pm)
    sc = sc.dropna(subset=["player_id"])
    sc = sc[["game_id", "team", "player_id", "offense_pct"]].drop_duplicates(["game_id", "player_id"])
    out = pg.merge(sc, on=["game_id", "team", "player_id"], how="outer")
    num = [c for c in out.columns if c not in ("game_id", "team", "player_id", "offense_pct")]
    out[num] = out[num].fillna(0)
    out = out.merge(team_tot, on=["game_id", "team"], how="inner")
    out = out.merge(games[["game_id", "season", "gameday"]], on="game_id", how="inner")
    pos = _positions()
    out["position"] = out["player_id"].map(pos)
    return out.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def _status_out(seasons) -> set:
    """(season, week, team, player) with final weekly status Out or Doubtful.

    Assumption (documented): final injury report status is known before kickoff;
    Doubtful players are treated as inactive.
    """
    inj = load("injuries", seasons).to_pandas()
    inj = inj[inj["report_status"].isin(["Out", "Doubtful"])]
    return set(zip(inj["season"], inj["week"], inj["team"], inj["gsis_id"]))


def build_roles(games: pd.DataFrame, player_games: pd.DataFrame) -> pd.DataFrame:
    """Pregame offensive roster per team-game: depth-chart players (rank within
    position) plus players active for the team in its previous game of the same
    season, minus
    Out/Doubtful players and players listed on another team's depth chart."""
    rows = []
    seasons = sorted(games["season"].unique())
    pos_map = _positions()
    old = [s for s in seasons if s <= 2024]
    dc_lookup = {}
    if old:
        dc = load("depth_charts", old).to_pandas()
        dc = dc[(dc["formation"] == "Offense") & dc["position"].isin(SKILL) & dc["week"].notna()
                & dc["gsis_id"].notna()]
        dc["rk"] = pd.to_numeric(dc["depth_team"], errors="coerce").fillna(9)
        dc = dc.sort_values(["rk", "depth_position"])
        for (s, w, t), x in dc.groupby(["season", "week", "club_code"]):
            x = x.drop_duplicates("gsis_id")
            x = x.assign(rank=x.groupby("position").cumcount() + 1)
            dc_lookup[(s, w, t)] = x[["gsis_id", "position", "rank"]].to_numpy()
    new = [s for s in seasons if s >= 2025]
    snaps_by_team = {}
    if new:
        dn = load("depth_charts", new).to_pandas()
        dn = dn[dn["pos_abb"].isin(SKILL) & dn["gsis_id"].notna()].copy()
        dn["ts"] = pd.to_datetime(dn["dt"], utc=True).dt.tz_localize(None)
        snaps_by_team = {t: x.sort_values("ts") for t, x in dn.groupby("team")}
    out_set = _status_out(seasons)

    prev_active = {}   # team -> (season, players active in the team's previous game)
    pg_by_game = {k: v for k, v in player_games.groupby("game_id")}
    for day, dg in games.groupby("gameday", sort=True):
        day_charts = {}
        for g in dg.itertuples():
            for t in (g.home_team, g.away_team):
                if g.season <= 2024:
                    arr = dc_lookup.get((g.season, g.week, t))
                else:
                    x = snaps_by_team.get(t)
                    arr = None
                    if x is not None:
                        x = x[x["ts"] < g.gameday]
                        if len(x):
                            snap = x[x["ts"] == x["ts"].max()].sort_values(["pos_rank", "pos_slot"])
                            snap = snap.drop_duplicates("gsis_id")
                            snap = snap.assign(rank=snap.groupby("pos_abb").cumcount() + 1)
                            arr = snap[["gsis_id", "pos_abb", "rank"]].to_numpy()
                day_charts[(g.game_id, t)] = arr
        listed_team = {}
        for (gid, t), arr in day_charts.items():
            if arr is not None:
                for pid in arr[:, 0]:
                    listed_team[pid] = t
        for g in dg.itertuples():
            for t in (g.home_team, g.away_team):
                arr = day_charts[(g.game_id, t)]
                seen = set()
                if arr is not None:
                    for pid, pos, rk in arr:
                        if (g.season, g.week, t, pid) in out_set:
                            continue
                        rows.append((g.game_id, t, pid, pos, int(rk), "depth_chart"))
                        seen.add(pid)
                prev_season, prev_ids = prev_active.get(t, (None, []))
                for pid in (prev_ids if prev_season == g.season else []):
                    if pid in seen or listed_team.get(pid, t) != t:
                        continue
                    if (g.season, g.week, t, pid) in out_set:
                        continue
                    pos = pos_map.get(pid)
                    if pos in SKILL:
                        rows.append((g.game_id, t, pid, pos, UNLISTED, "prev_game"))
        for g in dg.itertuples():
            x = pg_by_game.get(g.game_id)
            if x is None:
                continue
            for t, xt in x.groupby("team"):
                prev_active[t] = (g.season, list(xt["player_id"]))
    return pd.DataFrame(rows, columns=["game_id", "team", "player_id", "position", "rank", "src"])


def build_all(verbose=True):
    import polars as pl  # noqa: F401
    from .features import read_processed
    drives = read_processed("drives")
    games = read_processed("games")
    pbp = load("pbp")
    plays, pg, team_tot = build_play_tables(pbp, drives)
    player_games = build_player_games(pg, team_tot, games)
    roles = build_roles(games, player_games)
    plays.to_parquet(PROCESSED / "drive_plays.parquet", index=False)
    player_games.to_parquet(PROCESSED / "player_games.parquet", index=False)
    roles.to_parquet(PROCESSED / "roles.parquet", index=False)
    if verbose:
        print(f"drive_plays={len(plays)} player_games={len(player_games)} roles={len(roles)}")
    return plays, player_games, roles


# --------------------------------------------------------------------------- #
# Player model
# --------------------------------------------------------------------------- #
class Templates:
    """CSR view of drive_plays by drive_uid."""

    def __init__(self, plays: pd.DataFrame, n_drives: int):
        uid = plays["drive_uid"].to_numpy()
        self.kind = plays["kind"].to_numpy(np.int8)
        self.yards = plays["yards"].to_numpy(float)
        self.complete = plays["complete"].to_numpy(np.int8)
        self.intercept = plays["intercept"].to_numpy(np.int8)
        self.td = plays["td"].to_numpy(np.int8)
        counts = np.bincount(uid, minlength=n_drives)
        self.offsets = np.concatenate([[0], np.cumsum(counts)])


@dataclass
class TeamPlayers:
    team: str
    qb_id: str
    ids: list            # player ids (index = column in stat arrays)
    position: list
    rank: list
    tgt_share: np.ndarray
    catch: np.ndarray
    rec_bins: np.ndarray  # [K, NRB]
    rec_td_ratio: np.ndarray
    car_share: np.ndarray
    rush_bins: np.ndarray  # [K, NUB]
    rush_td_ratio: np.ndarray
    qb_idx: int
    notes: list
    p_active: np.ndarray = None   # pregame probability each player is active
    alpha_tgt: float | None = None
    alpha_car: float | None = None
    qb2_idx: int = -1             # backup QB (takes over after a simulated starter exit)


def _role_key(pos, rank):
    return (pos, min(int(rank), ROLE_CAP.get(pos, 1)) if rank != UNLISTED else UNLISTED)


class PlayerPriors:
    """Role-level averages estimated from games before the cutoff (last 2 seasons)."""

    def __init__(self, player_games: pd.DataFrame, roles: pd.DataFrame, cutoff, season):
        h = player_games[(player_games["gameday"] < cutoff) & (player_games["season"] >= season - 2)]
        x = h.merge(roles, on=["game_id", "team", "player_id"], how="inner", suffixes=("", "_r"))
        x["role"] = [_role_key(p, r) for p, r in zip(x["position_r"], x["rank"])]
        x["tshare"] = x["targets"] / x["team_targets"].clip(lower=1)
        x["cshare"] = x["carries"] / x["team_carries"].clip(lower=1)
        self.tshare = x.groupby("role")["tshare"].mean().to_dict()
        self.cshare = x.groupby("role")["cshare"].mean().to_dict()
        # P(active | pregame role): share of pregame-listed players who played.
        day = player_games.drop_duplicates("game_id").set_index("game_id")["gameday"]
        gs = player_games.drop_duplicates("game_id").set_index("game_id")["season"]
        ro = roles[roles["game_id"].map(day).lt(cutoff) & roles["game_id"].map(gs).ge(season - 2)]
        played = set(zip(player_games["game_id"], player_games["player_id"]))
        ro = ro.assign(role=[_role_key(p, r) for p, r in zip(ro["position"], ro["rank"])],
                       played=[(g, p_) in played for g, p_ in zip(ro["game_id"], ro["player_id"])])
        self.active = ro.groupby("role")["played"].mean().to_dict()
        hp = h.dropna(subset=["position"])
        self.pos = {}
        rbc = [f"rb{i}" for i in range(NRB)]
        ubc = [f"ub{i}" for i in range(NUB)]
        for pos, z in hp.groupby("position"):
            tg, rc, ca = z["targets"].sum(), z["receptions"].sum(), z["carries"].sum()
            self.pos[pos] = {
                "catch": rc / tg if tg > 50 else 0.62,
                "rec_bins": (z[rbc].sum().to_numpy() + 1) / (rc + NRB),
                "rec_td": (z["rec_td"].sum() + 1) / (rc + 20),
                "rush_bins": (z[ubc].sum().to_numpy() + 1) / (ca + NUB),
                "rush_td": (z["rush_td"].sum() + 1) / (ca + 30),
            }
        allz = hp
        self.default = {
            "catch": allz["receptions"].sum() / max(allz["targets"].sum(), 1),
            "rec_bins": (allz[rbc].sum().to_numpy() + 1) / (allz["receptions"].sum() + NRB),
            "rec_td": (allz["rec_td"].sum() + 1) / (allz["receptions"].sum() + 20),
            "rush_bins": (allz[ubc].sum().to_numpy() + 1) / (allz["carries"].sum() + NUB),
            "rush_td": (allz["rush_td"].sum() + 1) / (allz["carries"].sum() + 30),
        }

    def get(self, pos):
        return self.pos.get(pos, self.default)


def fit_team_players(game_id: str, team: str, qb_id: str | None, player_games: pd.DataFrame,
                     roles: pd.DataFrame, priors: PlayerPriors, cutoff, season,
                     p: PlayerParams | None = None, rng: np.random.Generator | None = None,
                     hist_by_player: dict | None = None, qb2_id: str | None = None) -> TeamPlayers:
    """Shares/efficiency for one team in one game from games strictly before `cutoff`.
    With `rng`, each player's game history is bootstrap-resampled (model uncertainty)."""
    p = p or PlayerParams()
    cutoff = pd.Timestamp(cutoff)
    ro = roles[(roles["game_id"] == game_id) & (roles["team"] == team)]
    notes = []
    if ro.empty:
        notes.append(f"{team}: no pregame roster/depth chart; using no skill players")
    ids = list(ro["player_id"])
    pos = list(ro["position"])
    rk = list(ro["rank"])
    if qb_id and qb_id not in ids:
        ids.append(qb_id); pos.append("QB"); rk.append(1)
        notes.append(f"{team}: expected QB {qb_id} not on depth chart; added as passer")
    if not qb_id:
        qbs = [i for i, x in zip(ids, pos) if x == "QB"]
        qb_id = qbs[0] if qbs else None
        notes.append(f"{team}: no expected QB; using {qb_id}")
    if qb2_id and qb2_id not in ids:
        ids.append(qb2_id); pos.append("QB"); rk.append(2)
    K = len(ids)
    # Team's recent games this season (for each player's recent activity)
    tg = player_games[(player_games["team"] == team) & (player_games["season"] == season)
                      & (player_games["gameday"] < cutoff)]
    recent_games = list(tg.drop_duplicates("game_id").sort_values("gameday")["game_id"])
    recent_games = recent_games[-p.active_window:]
    p_active = np.ones(K)
    tgt = np.zeros(K); catch = np.zeros(K); rtd = np.ones(K); utd = np.ones(K)
    car = np.zeros(K); rbins = np.zeros((K, NRB)); ubins = np.zeros((K, NUB))
    rbc = [f"rb{i}" for i in range(NRB)]
    ubc = [f"ub{i}" for i in range(NUB)]
    for i, (pid, ps, r) in enumerate(zip(ids, pos, rk)):
        role = _role_key(ps, r)
        pr = priors.get(ps)
        prior_t = priors.tshare.get(role, 0.02) if ps != "QB" else 0.0
        prior_c = priors.cshare.get(role, 0.02)
        if hist_by_player is not None:
            h = hist_by_player.get(pid)
            h = h[h["gameday"] < cutoff].tail(p.max_games) if h is not None else None
        else:
            h = player_games[(player_games["player_id"] == pid) & (player_games["gameday"] < cutoff)]
            h = h.tail(p.max_games)
        if h is not None and len(h) and rng is not None:
            h = h.iloc[rng.integers(0, len(h), len(h))].sort_values("gameday")
        r_role = priors.active.get(role, 0.8)
        if recent_games:
            played_ids = set(h["game_id"]) if h is not None else set()
            a_ = sum(g_ in played_ids for g_ in recent_games)
            p_active[i] = (a_ + p.k_active * r_role) / (len(recent_games) + p.k_active)
        else:
            p_active[i] = r_role
        if h is None or h.empty:
            w = np.zeros(0)
            h = None
        else:
            n = len(h)
            w = 0.5 ** (np.arange(n - 1, -1, -1) / p.halflife_games) * \
                p.season_carry ** (season - h["season"].to_numpy())

        def ws(col):
            return float((w * h[col].to_numpy()).sum()) if h is not None else 0.0

        if ps != "QB":
            tgt[i] = (ws("targets") + p.k_share * prior_t) / (ws("team_targets") + p.k_share)
        car[i] = (ws("carries") + p.k_share * prior_c) / (ws("team_carries") + p.k_share)
        tg, rc = ws("targets"), ws("receptions")
        catch[i] = (rc + p.k_catch * pr["catch"]) / (tg + p.k_catch)
        rb = np.array([ws(c) for c in rbc]) if h is not None else np.zeros(NRB)
        rbins[i] = (rb + p.k_bins * pr["rec_bins"]) / (rb.sum() + p.k_bins)
        rtd_rate = (ws("rec_td") + p.k_td * pr["rec_td"]) / (rc + p.k_td)
        rtd[i] = rtd_rate / pr["rec_td"]
        ub = np.array([ws(c) for c in ubc]) if h is not None else np.zeros(NUB)
        ubins[i] = (ub + p.k_bins * pr["rush_bins"]) / (ub.sum() + p.k_bins)
        ca = ws("carries")
        utd[i] = ((ws("rush_td") + p.k_td * pr["rush_td"]) / (ca + p.k_td)) / pr["rush_td"]
    # Only the expected starter takes QB snaps: other QBs get no designed runs.
    for i, (pid, ps) in enumerate(zip(ids, pos)):
        if ps == "QB" and pid != qb_id:
            car[i] = 0.0
    if tgt.sum() > 0:
        tgt = tgt / tgt.sum()
    if car.sum() > 0:
        car = car / car.sum()
    qb_idx = ids.index(qb_id) if qb_id in ids else -1
    if qb_idx >= 0:
        p_active[qb_idx] = 1.0   # the expected starter is assumed to play
    qb2_idx = ids.index(qb2_id) if qb2_id in ids and qb2_id != qb_id else -1
    return TeamPlayers(team, qb_id, ids, pos, rk, tgt, catch, rbins, rtd, car, ubins, utd,
                       qb_idx, notes, p_active, p.alpha_tgt, p.alpha_car, qb2_idx)


# --------------------------------------------------------------------------- #
# Play generation
# --------------------------------------------------------------------------- #
STATS = ["pass_att", "completions", "pass_yards", "pass_td", "interceptions", "sacks",
         "targets", "receptions", "rec_yards", "rec_td",
         "rush_att", "rush_yards", "rush_td",
         # big plays, from the same simulated plays (tail scanner, Milestone 4)
         "rec_long", "rush_long", "rec_20"]


@dataclass
class PlayerSim:
    teams: dict          # side (0 home, 1 away) -> TeamPlayers
    stats: dict          # side -> {stat: [n_sims, K] int array}
    team_plays: dict     # side -> {"completions", "pass_yards", "off_td_plays", ...: [n_sims]}
    n_sims: int
    active: dict = None  # side -> [n_sims, K] bool: player active in that simulation

    def player_stat(self, player_id: str, stat: str):
        """[n_sims] array, or None when the player is not on either pregame roster."""
        for side, tp in self.teams.items():
            if player_id in tp.ids:
                return self.stats[side][stat][:, tp.ids.index(player_id)]
        return None

    def player_active(self, player_id: str):
        """[n_sims] bool (False where the player sat out that simulation), or None."""
        for side, tp in self.teams.items():
            if player_id in tp.ids:
                return self.active[side][:, tp.ids.index(player_id)]
        return None

    def player_names(self):
        return {pid: side for side, tp in self.teams.items() for pid in tp.ids}


# Usage after a BLOWOUT starter exit: other starters sit too. Ratio of each pregame
# role's share after vs before the exit, 2018-2020 (scripts/blowout_starters.py;
# 124 team-games, 450 targets / 329 carries after the exit). Injury/benching exits
# showed no such change (ratios 0.84-1.16 for main roles), so they keep shares.
# Roles ranked 3+ (and unlisted players) use the rank-3 ratio.
BLOWOUT_TGT_RATIO = {"WR1": 0.600, "WR2": 0.763, "WR3": 1.509, "TE1": 0.957, "TE2": 0.824,
                     "TE3": 1.817, "RB1": 0.649, "RB2": 0.848, "RB3": 2.032}
BLOWOUT_CAR_RATIO = {"RB1": 0.211, "RB2": 1.044, "RB3": 2.402}


def _blowout_mult(tp, table: dict) -> np.ndarray:
    m = np.ones(len(tp.ids))
    for i, (ps, r) in enumerate(zip(tp.position, tp.rank)):
        key = f"{ps}{min(int(r), 3)}"
        m[i] = table.get(key, 1.0)
    return m


def _renorm(s: np.ndarray) -> np.ndarray:
    tot = s.sum(1, keepdims=True)
    return np.where(tot > 0, s / np.where(tot > 0, tot, 1), 0.0)


def _sample_players(share_rows: np.ndarray, eff: np.ndarray, cat: np.ndarray,
                    u: np.ndarray) -> np.ndarray:
    """One player per play. share_rows: [n_plays, K] shares in force for each play;
    eff: [K, C] likelihood of each play category per player. P(k) ∝ share * eff[k, cat]."""
    w = share_rows * eff[:, cat].T                     # [n_plays, K]
    tot = w.sum(1, keepdims=True)
    K = w.shape[1]
    w = np.where(tot > 0, w / np.where(tot > 0, tot, 1), 1.0 / K)
    c = np.cumsum(w, 1)
    return np.minimum((c < u[:, None]).sum(1), K - 1)


def _sim_shares(rng, share: np.ndarray, active: np.ndarray, alpha: float | None) -> np.ndarray:
    """Per-simulation shares among active players, optionally Dirichlet-dispersed."""
    s = share[None, :] * active                        # [n, K]
    if alpha is not None:
        conc = alpha * s / np.maximum(s.sum(1, keepdims=True), 1e-12)
        s = np.where(conc > 0, rng.gamma(np.maximum(conc, 1e-12)), 0.0)
    tot = s.sum(1, keepdims=True)
    return np.where(tot > 0, s / np.where(tot > 0, tot, 1), 0.0)


def simulate_players(sim, tmpl: Templates, teams: dict, rng: np.random.Generator) -> PlayerSim:
    n = len(sim.home_score)
    D = sim.drives
    uid = D["uid"]
    lens = tmpl.offsets[uid + 1] - tmpl.offsets[uid]
    drive_of = np.repeat(np.arange(len(uid)), lens)
    # index of each play inside the flat template arrays
    starts = np.repeat(tmpl.offsets[uid], lens)
    within = np.arange(len(drive_of)) - np.repeat(np.cumsum(lens) - lens, lens)
    pidx = starts + within
    kind = tmpl.kind[pidx]
    yards = tmpl.yards[pidx].copy()
    comp = tmpl.complete[pidx]
    intc = tmpl.intercept[pidx]
    td = tmpl.td[pidx]
    out = D["outcome"]
    is_td_drive = out == 0
    # Only keep template TD plays on simulated TD drives (a template is always a
    # drive with the same outcome, so this is a safety net).
    td = np.where(is_td_drive[drive_of], td, 0)

    # Rescale yards to the simulated field position.
    R = D["start_yl"].astype(float)
    S = np.bincount(drive_of, weights=yards, minlength=len(uid))
    f = np.ones(len(uid))
    tdd = is_td_drive & (S > 0)
    f[tdd] = R[tdd] / S[tdd]
    over = ~is_td_drive & (S > R - 1) & (S > 0)
    f[over] = np.maximum(R[over] - 1, 0) / S[over]
    yards = np.round(yards * f[drive_of])
    # TD drives: put rounding remainder on the TD play so the drive gains exactly R
    td_rows = np.flatnonzero(td == 1)
    if td_rows.size:
        S2 = np.bincount(drive_of, weights=yards, minlength=len(uid))
        d_ = drive_of[td_rows]
        yards[td_rows] += R[d_] - S2[d_]

    sim_of = D["sim"][drive_of]
    side_of = D["side"][drive_of]
    qbo = D["qb_out"][drive_of] if "qb_out" in D else np.zeros(len(drive_of), bool)
    qbb = D["qb_blowout"][drive_of] if "qb_blowout" in D else np.zeros(len(drive_of), bool)
    stats, team_plays, actives = {}, {}, {}
    for side in (0, 1):
        tp = teams[side]
        K = len(tp.ids)
        st = {s: np.zeros((n, K), dtype=np.int32) for s in STATS}
        pa = tp.p_active if tp.p_active is not None else np.ones(K)
        active = rng.random((n, K)) < pa[None, :]
        b2 = tp.qb2_idx
        if b2 >= 0:   # the backup plays in every simulation where the starter left
            exited = np.bincount(D["sim"][(D["side"] == side) & D.get("qb_out", np.zeros(len(uid), bool))],
                                 minlength=n) > 0
            active[:, b2] |= exited
        actives[side] = active
        tshare = _sim_shares(rng, tp.tgt_share, active, tp.alpha_tgt)
        cshare = _sim_shares(rng, tp.car_share, active, tp.alpha_car)
        tshare_b = _renorm(tshare * _blowout_mult(tp, BLOWOUT_TGT_RATIO)[None, :])
        cshare_b = _renorm(cshare * _blowout_mult(tp, BLOWOUT_CAR_RATIO)[None, :])
        m = side_of == side
        sk, sy, sc, si, std, ss = kind[m], yards[m], comp[m], intc[m], td[m], sim_of[m]
        q = tp.qb_idx
        so_ = qbo[m]
        sb_ = qbb[m]
        # QB on the field for each play: the backup after a simulated starter exit
        qi = np.where(so_ & (b2 >= 0), b2, q)
        if q >= 0:
            def addq(stat, mask, val=1):
                np.add.at(st[stat], (ss[mask], qi[mask]), val if np.isscalar(val) else val[mask])
            dropback = (sk == PASS) | (sk == PASS_NT)
            addq("pass_att", dropback)
            addq("completions", sc == 1)
            addq("pass_yards", sc == 1, sy)
            addq("pass_td", (sk == PASS) & (std == 1))
            addq("interceptions", si == 1)
            addq("sacks", sk == SACK)
            qr = (sk == SCRAM) | (sk == KNEEL)
            addq("rush_att", qr)
            addq("rush_yards", qr, sy)
            addq("rush_td", (sk == SCRAM) & (std == 1))
            np.maximum.at(st["rush_long"], (ss[qr], qi[qr]), sy[qr].astype(np.int32))
        # Targets
        tm = sk == PASS
        if tm.any() and K:
            eff = np.column_stack([1 - tp.catch]
                                  + [tp.catch * tp.rec_bins[:, b] for b in range(NRB)]
                                  + [tp.catch * tp.rec_bins[:, b] * tp.rec_td_ratio for b in range(NRB)])
            b = np.digitize(sy[tm], REC_BINS[1:-1])
            cat = np.where(sc[tm] == 1, 1 + b + NRB * std[tm], 0)
            rows_ = np.where(sb_[tm][:, None], tshare_b[ss[tm]], tshare[ss[tm]])
            r = _sample_players(rows_, eff, cat, rng.random(tm.sum()))
            s_ = ss[tm]
            np.add.at(st["targets"], (s_, r), 1)
            np.add.at(st["receptions"], (s_, r), sc[tm])
            np.add.at(st["rec_yards"], (s_, r), np.where(sc[tm] == 1, sy[tm], 0).astype(np.int32))
            np.add.at(st["rec_td"], (s_, r), std[tm])
            ry = np.where(sc[tm] == 1, sy[tm], 0).astype(np.int32)
            np.maximum.at(st["rec_long"], (s_, r), ry)
            np.add.at(st["rec_20"], (s_, r), (ry >= 20).astype(np.int32))
        # Designed runs
        rm = sk == RUN
        if rm.any() and K:
            eff = np.column_stack([tp.rush_bins[:, b] for b in range(NUB)]
                                  + [tp.rush_bins[:, b] * tp.rush_td_ratio for b in range(NUB)])
            b = np.digitize(sy[rm], RUSH_BINS[1:-1])
            cat = b + NUB * std[rm]
            rows_ = np.where(sb_[rm][:, None], cshare_b[ss[rm]], cshare[ss[rm]])
            r = _sample_players(rows_, eff, cat, rng.random(rm.sum()))
            if b2 >= 0:   # starter's designed runs go to the backup once he is out
                r = np.where(so_[rm] & (r == q), b2, r)
            s_ = ss[rm]
            np.add.at(st["rush_att"], (s_, r), 1)
            np.add.at(st["rush_yards"], (s_, r), sy[rm].astype(np.int32))
            np.add.at(st["rush_td"], (s_, r), std[rm])
            np.maximum.at(st["rush_long"], (s_, r), sy[rm].astype(np.int32))
        stats[side] = st
        team_plays[side] = {
            "completions": np.bincount(ss[sc == 1], minlength=n),
            "pass_yards": np.bincount(ss[sc == 1], weights=sy[sc == 1], minlength=n).astype(int),
            "off_td_plays": np.bincount(ss[std == 1], minlength=n),
            "td_drives": np.bincount(D["sim"][(D["side"] == side) & is_td_drive], minlength=n),
            "targets": np.bincount(ss[tm], minlength=n),
            "designed_runs": np.bincount(ss[rm], minlength=n),
            # game report: offensive plays (no penalties) and dropbacks incl. sacks/scrambles
            "plays": np.bincount(ss, minlength=n),
            "dropbacks": np.bincount(ss[np.isin(sk, [PASS, PASS_NT, SACK, SCRAM])], minlength=n),
        }
    return PlayerSim(teams, stats, team_plays, n, actives)


def read_tables():
    return (pd.read_parquet(PROCESSED / "drive_plays.parquet"),
            pd.read_parquet(PROCESSED / "player_games.parquet"),
            pd.read_parquet(PROCESSED / "roles.parquet"))


if __name__ == "__main__":
    build_all()
