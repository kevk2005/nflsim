"""Game report core (Milestone 4, tail scanner page): per player and stat, the
simulated distribution (mean, median, 10th/25th/75th/90th percentiles) and the
hit % at every threshold on a fine grid, next to the data behind it, plus the
walk-forward backtest that says which stats and threshold levels to trust.

Thresholds, levels, baseline window and the keep rule: config/tail_thresholds.toml
(the rule is fixed in SPEC.md, Milestone 4, before the backtest is run).

For each (game, player, stat, threshold):
  p_sim       share of simulations (player active) reaching the threshold, 95%
              Wilson Monte Carlo interval (bootstrap model range: gamereport.py)
  p_role      hit rate of players in the same pregame role (depth-chart position
              + rank) over the previous `prior_seasons` seasons, before the game
  p_hist      the player's own rate over his last `window_games` games played,
              shrunk toward p_role with `prior_games` of weight
  lift        p_sim - p_role
  p_market    info only: no-vig probability from a SAVED price at line t - 0.5
  actual      official result (nflverse weekly stats; longest plays from pbp)

CLI (all local, no Odds API calls):
  python -m nflsim.tails build-actuals
  python -m nflsim.tails backtest [--seasons 2024 2025 2026] [--sims 2000]
  python -m nflsim.tails report
The per-game report itself: python -m nflsim.gamereport build
"""
from __future__ import annotations

import argparse
import json
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .ingest import PROCESSED

CONFIG = Path(__file__).resolve().parents[2] / "config" / "tail_thresholds.toml"
ACTUALS = PROCESSED / "tail_actuals.parquet"
# report order; tds = rushing + receiving TDs
STAT_COLS = ["pass_yards", "pass_att", "completions", "pass_td", "interceptions", "rush_att", "rush_yards",
             "rush_long", "targets", "receptions", "rec_yards", "rec_long", "tds"]
STAT_LABELS = {"pass_yards": "pass yds", "pass_att": "pass att", "completions": "completions",
               "pass_td": "pass TDs", "interceptions": "INTs", "rush_att": "rush att", "rush_yards": "rush yds",
               "rush_long": "rush long", "targets": "targets", "receptions": "receptions",
               "rec_yards": "rec yds", "rec_long": "rec long", "tds": "TDs"}
LEVELS = ("low", "mid", "high")
PCTS = (10, 25, 50, 75, 90)


@dataclass(frozen=True)
class Threshold:
    stat: str
    at_least: int
    level: str
    positions: tuple
    market: str | None = None

    @property
    def name(self) -> str:
        return f"{self.stat}>={self.at_least}"

    @property
    def market_line(self) -> float | None:
        if self.market is None:
            return None
        return None if self.market == "anytime_td" else self.at_least - 0.5


def load_config(path: Path | None = None) -> dict:
    cfg = tomllib.loads((path or CONFIG).read_text())
    ths, stats = [], {}
    for s in cfg["stat"]:
        if s["stat"] not in STAT_COLS or s["stat"] in stats:
            raise ValueError(f"bad or repeated stat {s['stat']}")
        if not (s["from"] <= s["mid_from"] <= s["high_from"]) or s["step"] < 1:
            raise ValueError(f"bad grid for {s['stat']}")
        if s.get("market") == "anytime_td" and not (s["stat"] == "tds" and s["from"] == 1):
            raise ValueError("anytime_td prices only tds >= 1")
        grid = []
        for t in range(int(s["from"]), int(s["to"]) + 1, int(s["step"])):
            lv = "low" if t < s["mid_from"] else "mid" if t < s["high_from"] else "high"
            mk = s.get("market")
            if mk == "anytime_td" and t != 1:
                mk = None
            grid.append(Threshold(s["stat"], t, lv, tuple(s["positions"]), mk))
        stats[s["stat"]] = grid
        ths += grid
    cfg["thresholds"] = ths
    cfg["by_stat"] = stats
    return cfg


def stat_values(st: dict, k: int, stat: str) -> np.ndarray:
    """[n_sims] simulated values of a report stat for player column k."""
    if stat == "tds":
        return st["rush_td"][:, k] + st["rec_td"][:, k]
    return st[stat][:, k]


def report_position(pos: str) -> str:
    return "RB" if pos == "FB" else pos


# --------------------------------------------------------------------------- #
# Actual results (official stats + longest plays from play-by-play)
# --------------------------------------------------------------------------- #
def build_actuals(seasons=None) -> pd.DataFrame:
    """Per (game_id, player_id): STAT_COLS from official nflverse weekly stats
    (TDs = rushing + receiving + return/fumble-recovery, as books settle anytime TD)
    and longest reception / rush from play-by-play."""
    import polars as pl
    from .ingest import load
    games = pd.read_parquet(PROCESSED / "games.parquet")
    seasons = seasons or sorted(games["season"].unique())
    st = load("player_stats", list(seasons)).to_pandas()
    tdc = [c for c in ("rushing_tds", "receiving_tds", "special_teams_tds", "fumble_recovery_tds") if c in st]
    src = {"pass_yards": "passing_yards", "pass_att": "attempts", "completions": "completions",
           "pass_td": "passing_tds", "interceptions": "passing_interceptions", "rush_att": "carries",
           "rush_yards": "rushing_yards", "targets": "targets", "receptions": "receptions",
           "rec_yards": "receiving_yards"}
    out = pd.DataFrame({"game_id": st["game_id"], "player_id": st["player_id"],
                        **{k: st[v].fillna(0) for k, v in src.items()},
                        "tds": st[tdc].fillna(0).sum(axis=1)})
    p = load("pbp", list(seasons)).select(
        ["game_id", "play_type", "complete_pass", "yards_gained", "receiver_player_id", "rusher_player_id",
         "two_point_attempt"]).filter(pl.col("two_point_attempt").fill_null(0) == 0).to_pandas()
    rec = p[(p["play_type"] == "pass") & (p["complete_pass"] == 1) & p["receiver_player_id"].notna()]
    rec = rec.groupby(["game_id", "receiver_player_id"])["yards_gained"].max().rename("rec_long").reset_index() \
        .rename(columns={"receiver_player_id": "player_id"})
    rush = p[(p["play_type"] == "run") & p["rusher_player_id"].notna()]
    rush = rush.groupby(["game_id", "rusher_player_id"])["yards_gained"].max().rename("rush_long").reset_index() \
        .rename(columns={"rusher_player_id": "player_id"})
    out = out.merge(rec, on=["game_id", "player_id"], how="outer").merge(rush, on=["game_id", "player_id"], how="outer")
    out[STAT_COLS] = out[STAT_COLS].fillna(0)
    out = out.merge(games[["game_id", "season", "gameday"]], on="game_id", how="inner")
    out.to_parquet(ACTUALS, index=False)
    return out


def load_actuals() -> pd.DataFrame:
    if ACTUALS.exists():
        a = pd.read_parquet(ACTUALS)
        if set(STAT_COLS) <= set(a.columns):
            return a
    return build_actuals()


# --------------------------------------------------------------------------- #
# Role baseline and player history
# --------------------------------------------------------------------------- #
def role_name(pos, rank) -> str | None:
    from .players import _role_key
    return "%s%s" % _role_key(pos, rank) if isinstance(pos, str) else None


class Baseline:
    """History = games the player played (snap counts / usage, as elsewhere).
    Roles = pregame depth-chart position + rank (players.build_roles)."""

    def __init__(self, actuals: pd.DataFrame, player_games: pd.DataFrame, roles: pd.DataFrame,
                 thresholds: list, cfg: dict):
        played = player_games[["game_id", "player_id", "team"]].drop_duplicates(["game_id", "player_id"])
        h = played.merge(actuals[["game_id", "player_id"] + STAT_COLS], on=["game_id", "player_id"], how="left")
        h = h.merge(player_games.drop_duplicates("game_id")[["game_id", "season", "gameday"]],
                    on="game_id", how="left")
        h[STAT_COLS] = h[STAT_COLS].fillna(0)
        self.names = [t.name for t in thresholds]
        hits = pd.DataFrame({t.name: (h[t.stat].to_numpy() >= t.at_least).astype(np.float32)
                             for t in thresholds}, index=h.index)
        h = pd.concat([h, hits], axis=1)
        r = roles.drop_duplicates(["game_id", "player_id"])
        h = h.merge(r[["game_id", "player_id", "position", "rank"]], on=["game_id", "player_id"], how="left")
        h["role"] = [role_name(p, k) for p, k in zip(h["position"], h["rank"])]
        self.h = h.sort_values("gameday")
        self.by_player = {k: v for k, v in self.h.groupby("player_id")}
        self.window = int(cfg["baseline"]["window_games"])
        self.k = float(cfg["baseline"]["prior_games"])
        self.prior_seasons = int(cfg["baseline"]["prior_seasons"])
        self.role_of = {(g, p): role_name(a, b) for g, p, a, b in
                        zip(r["game_id"], r["player_id"], r["position"], r["rank"])}
        self._prior = {}

    def prior(self, cutoff, season):
        key = (pd.Timestamp(cutoff), int(season))
        if key not in self._prior:
            x = self.h[(self.h["gameday"] < cutoff) & (self.h["season"] >= season - self.prior_seasons)]
            xr = x.dropna(subset=["role"])
            by_role, n_role = xr.groupby("role")[self.names].mean(), xr.groupby("role").size()
            by_pos = x.dropna(subset=["position"]).groupby("position")[self.names].mean()
            self._prior[key] = (by_role, by_pos, x[self.names].mean(), n_role)
        return self._prior[key]

    def role(self, player_id, game_id, position) -> str:
        return self.role_of.get((game_id, player_id)) or f"{position}?"

    def role_rates(self, player_id, game_id, position, cutoff, season) -> tuple:
        """(role key, {threshold: role rate}, n role player-games behind it)."""
        by_role, by_pos, overall, n_role = self.prior(cutoff, season)
        role = self.role_of.get((game_id, player_id))
        if role in by_role.index:
            return role, by_role.loc[role], int(n_role[role])
        if position in by_pos.index:
            return f"{position} (any rank)", by_pos.loc[position], 0
        return "all players", overall, 0

    def history(self, player_id, cutoff) -> pd.DataFrame:
        """His last `window` games played before the cutoff (oldest first)."""
        x = self.by_player.get(player_id)
        return x[x["gameday"] < cutoff].tail(self.window) if x is not None else self.h.iloc[:0]

    def get(self, player_id, game_id, position, cutoff, season) -> tuple:
        """({threshold: shrunk own rate}, n games in the window, {threshold: role rate}, role key)."""
        role, pr, _ = self.role_rates(player_id, game_id, position, cutoff, season)
        x = self.history(player_id, cutoff)
        n = len(x)
        hits = x[self.names].sum() if n else pd.Series(0.0, index=self.names)
        return ((hits + self.k * pr) / (n + self.k)).to_dict(), n, pr.to_dict(), role


# --------------------------------------------------------------------------- #
# Market (info only): saved prices
# --------------------------------------------------------------------------- #
MARKET_STAT = {"passing_yards": "pass_yards", "pass_attempts": "pass_att", "completions": "completions",
               "passing_tds": "pass_td", "rush_attempts": "rush_att", "rushing_yards": "rush_yards",
               "receptions": "receptions", "receiving_yards": "rec_yards", "anytime_td": "tds"}


def market_table(game_ids=None, as_of=None) -> pd.DataFrame:
    """Per (game, player, market, line): median no-vig probability across books
    (vig-included implied if only one side is saved), best price and its book.
    Latest pre-kickoff snapshot per (game, book, market) only."""
    from .odds import american_to_decimal, american_to_prob
    path = PROCESSED / "prop_snapshots.parquet"
    cols = ["game_id", "player_id", "market", "line", "p_market", "one_sided", "best_price", "best_book", "n_books"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    p = pd.read_parquet(path)
    p = p[p["player_id"].notna() & p["side"].isin(["over", "yes"])]
    if game_ids is not None:
        p = p[p["game_id"].isin(set(game_ids))]
    p = p.assign(t=p["snapshot_ts"].fillna(p["pulled_at"]))
    p = p[p["t"] < p["commence_time"]]
    if as_of is not None:
        p = p[p["t"] <= as_of]
    if p.empty:
        return pd.DataFrame(columns=cols)
    p = p[p["t"] == p.groupby(["game_id", "book", "market_key"])["t"].transform("max")]
    p = p.assign(line=p["point"].where(p["market"] != "anytime_td", np.nan),
                 implied=american_to_prob(p["price"].to_numpy(float)),
                 dec=american_to_decimal(p["price"].to_numpy(float)))
    p["prob"] = p["fair_prob"].fillna(p["implied"])
    p["two"] = p["fair_prob"].notna()
    rows = []
    for (g, pid, m, ln), x in p.groupby(["game_id", "player_id", "market", p["line"].fillna(-1)]):
        two = x[x["two"]]
        use = two if len(two) else x
        b = x.loc[x["dec"].idxmax()]
        rows.append({"game_id": g, "player_id": pid, "market": m, "line": None if ln == -1 else float(ln),
                     "p_market": float(use["prob"].median()), "one_sided": not len(two),
                     "best_price": float(b["price"]), "best_book": b["book"], "n_books": int(x["book"].nunique())})
    return pd.DataFrame(rows, columns=cols)


def _market_lookup(mt: pd.DataFrame):
    idx = {}
    for r in mt.itertuples():
        idx[(r.game_id, r.player_id, r.market, -1.0 if r.line is None or pd.isna(r.line) else float(r.line))] = r

    def get(game_id, player_id, t: Threshold):
        if t.market is None:
            return None
        ln = -1.0 if t.market_line is None else float(t.market_line)
        return idx.get((game_id, player_id, t.market, ln))
    return get


# --------------------------------------------------------------------------- #
# Per-player rows of one simulated game
# --------------------------------------------------------------------------- #
def wilson(k, n, z=1.959964):
    """Vectorized 95% Wilson interval."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return np.minimum(np.maximum(0.0, c - h), p), np.maximum(np.minimum(1.0, c + h), p)   # float noise at k=0, n


def hit_probs(x: np.ndarray, ts) -> np.ndarray:
    """P(X >= t) for each t, from samples x."""
    xs = np.sort(x)
    return (len(xs) - np.searchsorted(xs, np.asarray(ts, float), "left")) / len(xs)


def pit(x: np.ndarray, y: float, u: float) -> float:
    """Randomized PIT: P(X < y) + u * P(X = y)."""
    return float((x < y).mean() + u * (x == y).mean())


def tracked_players(sg, min_active: int):
    """(side, k, TeamPlayers, usage role) for every player the sim tracks."""
    from .backtest import _roles_from_model
    for side, tp in sg.players.teams.items():
        usage = _roles_from_model(tp)
        act = sg.players.active[side]
        for k in range(len(tp.ids)):
            if act[:, k].sum() >= min_active:
                yield side, k, tp, usage.get(k, "")


def game_rows(sg, cfg, base: Baseline, cutoff, season, mlook=None, actual=None, played=None,
              rng: np.random.Generator | None = None) -> tuple:
    """(threshold rows, stat rows) for every tracked player of one simulated game.
    With `actual`/`played`, rows carry the result (y, PIT) for grading."""
    kr = cfg["keep_rule"]
    trows, srows = [], []
    rng = rng or np.random.default_rng(0)
    for side, k, tp, usage in tracked_players(sg, cfg["report"]["min_active_sims"]):
        st, act = sg.players.stats[side], sg.players.active[side][:, k]
        pid, pos = tp.ids[k], report_position(tp.position[k])
        hist, n_hist, role_r, role = base.get(pid, sg.game_id, tp.position[k], cutoff, season)
        a = None if actual is None or played is None or pid not in played else actual.get(pid, {})
        for stat, grid in cfg["by_stat"].items():
            if pos not in grid[0].positions:
                continue
            x = stat_values(st, k, stat)[act]
            if not x.any():
                continue                         # zero in every active simulation: left out
            q = np.percentile(x, PCTS, method="inverted_cdf")
            nz = float((x > 0).mean())
            min_nz = kr["pit_min_nonzero_tds"] if stat == "tds" else kr["pit_min_nonzero"]
            srow = {"game_id": sg.game_id, "team": tp.team, "player_id": pid, "position": pos, "role": role,
                    "usage_role": usage, "stat": stat, "n_active_sims": int(len(x)),
                    "p_active": float(act.mean()), "mean": float(x.mean()),
                    **{f"p{p}": float(v) for p, v in zip(PCTS, q)}, "p_nonzero": nz,
                    "pit_scored": bool(act.mean() >= kr["pit_min_active"] and nz >= min_nz)}
            ps = hit_probs(x, [t.at_least for t in grid])
            lo, hi = wilson(np.round(ps * len(x)), len(x))
            yv = None if a is None else float(a.get(stat, 0) or 0)
            if actual is not None:
                srow.update({"played": a is not None, "actual_value": yv,
                             "pit": None if a is None else pit(x, yv, float(rng.random()))})
            srows.append(srow)
            for t, p_, l_, h_ in zip(grid, ps, lo, hi):
                row = {"game_id": sg.game_id, "team": tp.team, "player_id": pid, "position": pos, "role": role,
                       "stat": stat, "at_least": t.at_least, "level": t.level, "p_sim": float(p_),
                       "mc_lo": float(l_), "mc_hi": float(h_), "n_active_sims": int(len(x)),
                       "p_role": float(role_r[t.name]), "p_hist": float(hist[t.name]), "n_hist_games": n_hist}
                row["lift"] = row["p_sim"] - row["p_role"]
                if mlook is not None:
                    m = mlook(sg.game_id, pid, t)
                    row.update({"p_market": None if m is None else m.p_market,
                                "market_one_sided": None if m is None else bool(m.one_sided),
                                "best_price": None if m is None else m.best_price,
                                "n_books": None if m is None else m.n_books})
                if actual is not None:
                    row["played"] = a is not None
                    row["y"] = None if a is None else float(yv >= t.at_least)
                trows.append(row)
    return trows, srows


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def best_config() -> dict:
    p = PROCESSED / "prop_wf_best.json"
    return json.loads(p.read_text()) if p.exists() else {"mode": "independent", "qb_exits": "on"}


def shared_inputs(cfg):
    from .pipeline import load_context
    ctx = load_context()
    return ctx, Baseline(load_actuals(), ctx.player_games, ctx.roles, cfg["thresholds"], cfg)


def tail_walk_forward(seasons=(2024, 2025, 2026), n_sims: int = 2000, seed: int = 0, mode: str | None = None,
                      qb_exits: bool | None = None, cfg: dict | None = None, verbose: bool = True,
                      save: bool = True, game_ids=None) -> tuple:
    """Weekly walk-forward: the team model is refit on data before each week's
    first game; player usage, baselines and role averages use games before it.
    Default config = the Milestone 3 kept one (prop_wf_best.json); pass shock off.
    Returns (threshold rows, stat rows); saves tail_wf.parquet / tail_pit.parquet."""
    from .team_model import fit_team_model
    from .pipeline import simulate_game
    cfg = cfg or load_config()
    best = best_config()
    mode = mode or best["mode"]
    qb_exits = (best["qb_exits"] == "on") if qb_exits is None else qb_exits
    ctx, base = shared_inputs(cfg)
    acts = load_actuals()
    act_by_game = {g: {r["player_id"]: r for r in x.to_dict("records")} for g, x in acts.groupby("game_id")}
    played_by_game = {g: set(x["player_id"]) for g, x in ctx.player_games.groupby("game_id")}
    games = ctx.games
    test = games[games["season"].isin(seasons) & games["label_home_score"].notna()]
    if game_ids is not None:
        test = test[test["game_id"].isin(set(game_ids))]
    rng = np.random.default_rng(seed)
    trows, srows, t0 = [], [], time.time()
    for (season, week), wk in test.groupby(["season", "week"], sort=True):
        cutoff = wk["gameday"].min()
        tm = fit_team_model(cutoff, ctx.drives, ctx.pat, ctx.gf)
        for g in wk.itertuples():
            sg = simulate_game(ctx, g.game_id, n_sims, seed=int(rng.integers(1 << 31)), cutoff=cutoff,
                               team_model=tm, mode=mode, qb_exits=qb_exits, pass_shock=0.0)
            tr, sr = game_rows(sg, cfg, base, cutoff, int(season), actual=act_by_game.get(g.game_id, {}),
                               played=played_by_game.get(g.game_id, set()),
                               rng=np.random.default_rng(int(rng.integers(1 << 31))))
            trows.append(pd.DataFrame(tr).assign(season=int(season), week=int(week)))
            srows.append(pd.DataFrame(sr).assign(season=int(season), week=int(week)))
        if verbose:
            print(f"{season} wk{week:>2}: {len(wk)} games, {time.time() - t0:.0f}s", flush=True)
    td, sd = pd.concat(trows, ignore_index=True), pd.concat(srows, ignore_index=True)
    for d in (td, sd):
        d["mode"], d["qb_exits"] = mode, qb_exits
    if save:
        td.to_parquet(PROCESSED / "tail_wf.parquet", index=False)
        sd.to_parquet(PROCESSED / "tail_pit.parquet", index=False)
    return td, sd


# --------------------------------------------------------------------------- #
# Report + keep test
# --------------------------------------------------------------------------- #
def _boot_mean_by_game(values, games, B: int = 1000, seed: int = 0):
    """95% CI of a mean, resampling games."""
    rng = np.random.default_rng(seed)
    g = pd.DataFrame({"v": np.asarray(values, float), "g": np.asarray(games)}).groupby("g")["v"].agg(["sum", "size"])
    s, n = g["sum"].to_numpy(), g["size"].to_numpy()
    idx = rng.integers(0, len(g), (B, len(g)))
    bs = s[idx].sum(1) / n[idx].sum(1)
    return [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def calibration(d: pd.DataFrame, kr: dict, ci: bool = True) -> dict:
    """Calibration over the pre-registered bands; scored bands need n >= min."""
    edges = np.asarray(kr["calibration_bands"], float)
    band = np.clip(np.searchsorted(edges, d["p_sim"].to_numpy(float), "right") - 1, 0, len(edges) - 2)
    bands, ok = [], True
    for i, x in d.groupby(band):
        n = len(x)
        r = {"band": f"{100 * edges[i]:.0f}–{100 * edges[i + 1]:.0f}%", "n": int(n),
             "pred": float(x["p_sim"].mean()), "actual": float(x["y"].mean())}
        r["err"] = r["pred"] - r["actual"]
        r["scored"] = bool(n >= kr["calibration_min_band_n"])
        r["pass"] = bool(abs(r["err"]) <= kr["calibration_tolerance"]) if r["scored"] else None
        if ci and r["scored"]:
            r["actual_ci95"] = _boot_mean_by_game(x["y"], x["game_id"])
        ok &= r["pass"] is not False
        bands.append(r)
    scored = [r for r in bands if r["scored"]]
    return {"n": int(len(d)), "bands": bands, "n_scored_bands": len(scored),
            "max_abs_err_scored": max((abs(r["err"]) for r in scored), default=None),
            "pass": bool(ok) if scored else None}


def percentile_check(s: pd.DataFrame, kr: dict) -> dict:
    tol = kr["percentile_tolerance"]
    x = s[s["pit_scored"]]
    a = s["pit"].dropna().astype(float)
    all_rows = {"n": int(len(a)), "below_p10": float((a < .1).mean()) if len(a) else None,
                "below_p50": float((a < .5).mean()) if len(a) else None,
                "above_p90": float((a > .9).mean()) if len(a) else None}
    if x.empty:
        return {"n": 0, "pass": None, "all_rows": all_rows}
    p = x["pit"].astype(float)
    out = {"n": int(len(x)), "n_games": int(x["game_id"].nunique())}
    for k, m in (("below_p10", p < .1), ("below_p50", p < .5), ("above_p90", p > .9)):
        out[k] = float(m.mean())
        out[k + "_ci95"] = _boot_mean_by_game(m.astype(float), x["game_id"])
    out["pass"] = bool(abs(out["below_p10"] - .1) <= tol and abs(out["below_p50"] - .5) <= tol
                       and abs(out["above_p90"] - .1) <= tol)
    out["all_rows"] = all_rows     # reported only: every graded row, bench players included
    return out


def _brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def tail_report(td: pd.DataFrame, sd: pd.DataFrame, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    kr = cfg["keep_rule"]
    d = td[td["played"].astype(bool)].copy()
    d["y"] = d["y"].astype(float)
    s = sd[sd["played"].astype(bool)].copy()
    out = {"n_rows": int(len(td)), "n_graded": int(len(d)), "n_games": int(d["game_id"].nunique()),
           "n_stat_rows_graded": int(len(s)), "void_rate": float(1 - td["played"].astype(bool).mean()),
           "config": {"mode": str(td["mode"].iloc[0]), "qb_exits": bool(td["qb_exits"].iloc[0]),
                      "pass_shock": "off"}}
    out["by_level"] = {lv: calibration(d[d["level"] == lv], kr) for lv in LEVELS if (d["level"] == lv).any()}
    out["by_stat"] = {}
    for stat in STAT_COLS:
        x = d[d["stat"] == stat]
        if x.empty:
            continue
        r = calibration(x, kr)
        r.update({"brier_sim": _brier(x["p_sim"], x["y"]), "brier_role": _brier(x["p_role"], x["y"]),
                  "brier_hist": _brier(x["p_hist"], x["y"])})
        r["percentile"] = percentile_check(s[s["stat"] == stat], kr)
        r["by_level"] = {lv: calibration(x[x["level"] == lv], kr, ci=False)
                         for lv in LEVELS if (x["level"] == lv).any()}
        out["by_stat"][stat] = r
    out["percentile_overall"] = percentile_check(s, kr)
    a_ok = all(v["pass"] for v in out["by_level"].values())
    b_ok = bool(out["percentile_overall"]["pass"])
    out["keep_test"] = {
        "a_calibration_by_level": {lv: ("PASS" if v["pass"] else "FAIL" if v["pass"] is False else "NO DATA")
                                   for lv, v in out["by_level"].items()},
        "b_percentiles_pooled": "PASS" if b_ok else "FAIL",
        "overall": "PASS" if a_ok and b_ok else "FAIL"}
    out["trust"] = trust_table(out)
    return out


def trust_table(rep: dict) -> dict:
    """{stat: {level: label}}: 'trusted', 'not trusted: <reason>' or 'not enough data ...'."""
    out = {}
    for stat, r in rep.get("by_stat", {}).items():
        pc = r.get("percentile", {})
        out[stat] = {}
        for lv, c in r.get("by_level", {}).items():
            if c["pass"] is None:
                out[stat][lv] = "not enough data (no band with n ≥ 200)"
                continue
            why = []
            if c["pass"] is False:
                bad = [b for b in c["bands"] if b["pass"] is False]
                w = max(bad, key=lambda b: abs(b["err"]))
                why.append(f"calibration off in {len(bad)} band(s), worst {w['band']}: sim {100 * w['pred']:.1f}% "
                           f"vs actual {100 * w['actual']:.1f}%")
            if pc.get("pass") is False:
                why.append(f"percentiles off (below p10 {100 * pc['below_p10']:.0f}%, below median "
                           f"{100 * pc['below_p50']:.0f}%, above p90 {100 * pc['above_p90']:.0f}%)")
            out[stat][lv] = "trusted" if not why else "not trusted: " + "; ".join(why)
    return out


def trust_labels(rep: dict | None):
    """Function (stat, level) -> label for the app, from the backtest report."""
    tt = (rep or {}).get("trust", {})

    def label(stat, level):
        if rep is None:
            return "not trusted: no backtest"
        return tt.get(stat, {}).get(level, "not trusted: not in the backtest")
    return label


def load_report() -> dict | None:
    """The saved backtest report, or None (also for a report from the retired flag-based version)."""
    p = PROCESSED / "tail_report.json"
    rep = json.loads(p.read_text()) if p.exists() else None
    return rep if rep is not None and "trust" in rep else None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m nflsim.tails")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-actuals")
    b = sub.add_parser("backtest")
    b.add_argument("--seasons", type=int, nargs="*", default=[2024, 2025, 2026])
    b.add_argument("--sims", type=int, default=2000)
    sub.add_parser("report")
    a = ap.parse_args(argv)
    if a.cmd == "build-actuals":
        d = build_actuals()
        print(f"tail_actuals: {len(d)} player-games")
        return
    if a.cmd == "backtest":
        tail_walk_forward(tuple(a.seasons), a.sims)
    td = pd.read_parquet(PROCESSED / "tail_wf.parquet")
    sd = pd.read_parquet(PROCESSED / "tail_pit.parquet")
    rep = tail_report(td, sd)
    (PROCESSED / "tail_report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({"keep_test": rep["keep_test"], "trust": rep["trust"]}, indent=1))


if __name__ == "__main__":
    main()
