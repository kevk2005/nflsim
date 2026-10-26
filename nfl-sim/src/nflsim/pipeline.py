"""Glue: load everything once, then simulate a game with team and player layers."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .backtest import load_inputs
from .players import (PlayerPriors, PlayerSim, Templates, fit_team_players, read_tables,
                      simulate_players)
from .simulator import SimResult, matchup_from_features, simulate
from .team_model import TeamModel, fit_team_model


@dataclass
class Context:
    games: pd.DataFrame
    gf: pd.DataFrame
    drives: pd.DataFrame
    pat: pd.DataFrame
    tmpl: Templates
    player_games: pd.DataFrame
    roles: pd.DataFrame
    hist_by_player: dict
    _priors: dict = field(default_factory=dict)

    def priors(self, cutoff, season) -> PlayerPriors:
        key = (pd.Timestamp(cutoff), int(season))
        if key not in self._priors:
            self._priors[key] = PlayerPriors(self.player_games, self.roles, cutoff, season)
        return self._priors[key]


def load_context() -> Context:
    games, gf, drives, pat = load_inputs()
    plays, pg, roles = read_tables()
    tmpl = Templates(plays, int(drives["drive_uid"].max()) + 1)
    hist = {k: v for k, v in pg.groupby("player_id")}
    return Context(games, gf, drives, pat, tmpl, pg, roles, hist)


@dataclass
class SimulatedGame:
    game_id: str
    home_team: str
    away_team: str
    gameday: pd.Timestamp
    n_sims: int
    sim: SimResult
    players: PlayerSim
    team_model: TeamModel
    missing: list
    feature_row: pd.Series
    anchor: dict = None          # mode + line + solved offense shifts (see anchor.py)


def simulate_game(ctx: Context, game_id: str, n_sims: int = 10000, seed: int = 0,
                  cutoff=None, team_model: TeamModel | None = None,
                  bootstrap_rng: np.random.Generator | None = None,
                  mode: str = "independent", home_spread: float | None = None,
                  total: float | None = None, qb_exits: bool = True,
                  as_of: pd.Timestamp | None = None, pass_shock: float = 0.0) -> SimulatedGame:
    """Simulate one game. Models are fit on data dated before `cutoff`
    (default: the game day). With `bootstrap_rng`, the drive model and every
    player's usage history are bootstrap-resampled (one model-uncertainty draw).

    mode="market_anchored" shifts each team's offense so the simulated mean
    margin/total match the market (anchor.py). home_spread (sportsbook style,
    negative = home favored) and total override the nflverse line. If no
    complete line exists the game is simulated in independent mode and the
    fallback is recorded in `anchor` and `missing`.

    qb_exits=False turns off simulated starting-QB early exits (blowout pulls and
    injury/benching); the starter then plays every snap.

    pass_shock: sd of the shared per-game passing shock (pass-share tilt of the
    drive templates, one draw per team per simulation); 0 = off. The CLI switch
    `--pass-shock on` uses simulator.PASS_SHOCK_SD.

    as_of: latest time an odds snapshot may come from (default: now). Snapshots
    from after kickoff are never used, whatever as_of is."""
    from .anchor import MODES, describe, get_line, solve_anchor
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    rows = ctx.gf[ctx.gf["game_id"] == game_id]
    if rows.empty:
        raise KeyError(f"unknown game_id {game_id}")
    row = rows.iloc[0]
    cutoff = pd.Timestamp(cutoff if cutoff is not None else row["gameday"])
    if team_model is None:
        team_model = fit_team_model(cutoff, ctx.drives, ctx.pat, ctx.gf,
                                    rng=bootstrap_rng, bootstrap=bootstrap_rng is not None)
    pri = ctx.priors(cutoff, row["season"])
    teams = {}
    for side, t in ((0, row["home_team"]), (1, row["away_team"])):
        side_name = "home" if side == 0 else "away"
        qb2 = row.get(f"{side_name}_qb2_id")
        teams[side] = fit_team_players(game_id, t, row[f"{side_name}_qb_id"], ctx.player_games,
                                       ctx.roles, pri, cutoff, int(row["season"]),
                                       rng=bootstrap_rng, hist_by_player=ctx.hist_by_player,
                                       qb2_id=qb2 if isinstance(qb2, str) else None)
    mu = replace(matchup_from_features(row), qb_exits=bool(qb_exits), pass_shock=float(pass_shock))
    info = {"mode": "independent"}
    if mode == "market_anchored":
        grow = ctx.games[ctx.games["game_id"] == game_id].iloc[0]
        line = get_line(grow, home_spread, total, snapshot=odds_snapshot_for(game_id, grow, as_of))
        if line.complete:
            a = solve_anchor(team_model, mu, line.target_margin, line.target_total, n_sims, seed)
            sim = a.sim
            info = {"mode": "market_anchored", "target_margin": line.target_margin,
                    "target_total": line.target_total, "source": line.source, "as_of": line.as_of,
                    "warning": line.warning,
                    "d_home": a.d_home, "d_away": a.d_away, "converged": a.converged,
                    "iterations": a.iterations, "err_margin": a.err_margin,
                    "err_total": a.err_total, "base_margin": a.base_margin,
                    "base_total": a.base_total, "sim_margin": float(sim.margin.mean()),
                    "sim_total": float(sim.total.mean())}
            from .anchor import ep_change
            info["ep_home"] = ep_change(team_model, mu, 0, a.d_home)
            info["ep_away"] = ep_change(team_model, mu, 1, a.d_away)
        else:
            info = {"mode": "independent", "fallback": f"{line.note} ({line.source})"}
            sim = simulate(team_model, mu, n_sims, seed=seed)
    else:
        sim = simulate(team_model, mu, n_sims, seed=seed)
    ps = simulate_players(sim, ctx.tmpl, teams, np.random.default_rng([seed, 1]))
    missing = [
        "Weather: forecasts not available; game-time conditions are never used.",
        "Injuries: final weekly report status assumed known before kickoff; "
        "Out and Doubtful treated as inactive; Questionable treated as active.",
        "Inactives: official game-day inactive lists are not used.",
        ("QB early exits: simulated (blowout pulls and injury/benching, rates from 2018-20)."
         if qb_exits else "QB early exits: OFF (the starter plays every snap)."),
        (f"Per-game passing shock: ON (sd {pass_shock:g})." if pass_shock else
         "Per-game passing shock: OFF."),
        "Schemes / coaching changes: not modeled.",
        ("Market lines: used as the anchor (mean margin and total), never as a model input."
         if info["mode"] == "market_anchored" else
         "Market lines: not used (independent mode); never a model input."),
        "Return touchdowns are not credited to any player (anytime TD undercounts them).",
        f"Starting QBs (pregame estimate): {row['home_team']} {row['home_qb_id']} "
        f"[{row['home_qb_source']}], {row['away_team']} {row['away_qb_id']} [{row['away_qb_source']}].",
    ]
    for tp in teams.values():
        missing += tp.notes
    missing = describe(info, row["home_team"], row["away_team"]) + missing
    return SimulatedGame(game_id, row["home_team"], row["away_team"], row["gameday"], n_sims,
                         sim, ps, team_model, missing, row, info)


def odds_snapshot_for(game_id: str, game_row: pd.Series, as_of=None) -> dict | None:
    """Latest pre-kickoff odds-feed consensus for a game, or None."""
    from . import odds_feed as F
    sched = F.load_schedule()
    k = sched.loc[sched.game_id == game_id, "kickoff_utc"]
    if k.empty or pd.isna(k.iloc[0]):
        return None
    kickoff = k.iloc[0]
    ref = pd.Timestamp(as_of) if as_of is not None else F.utcnow()
    snap = F.latest_consensus(game_id, kickoff, as_of=ref)
    if snap is not None:
        snap["age_ref"] = "kickoff" if kickoff <= ref else "now"
    return snap


def player_table(game: SimulatedGame, names: dict | None = None) -> pd.DataFrame:
    """Mean simulated box score per active player."""
    out = []
    for side, tp in game.players.teams.items():
        st = game.players.stats[side]
        for k, pid in enumerate(tp.ids):
            r = {"team": tp.team, "player_id": pid, "name": (names or {}).get(pid, pid),
                 "position": tp.position[k], "rank": tp.rank[k],
                 "tgt_share": tp.tgt_share[k], "car_share": tp.car_share[k]}
            for s, a in st.items():
                r[s] = a[:, k].mean()
            out.append(r)
    return pd.DataFrame(out)


def player_names() -> dict:
    from .ingest import load
    r = load("rosters").to_pandas()[["gsis_id", "full_name", "season"]].dropna()
    r = r.sort_values("season").drop_duplicates("gsis_id", keep="last")
    return dict(zip(r["gsis_id"], r["full_name"]))
