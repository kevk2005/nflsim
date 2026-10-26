"""Weekly walk-forward evaluation (chronological only).

For each (season, week): fit the TeamModel on data dated before that week's
first game, simulate every game that week, and record predictions. Compare
against a home-team baseline, Elo, and the closing market line (benchmark only).

Run: python -m nflsim.backtest --start 2021 --sims 4000
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from .features import PROCESSED, read_processed
from .simulator import matchup_from_features, simulate
from .anchor import get_line, solve_anchor
from .team_model import compute_elo, fit_team_model, with_elo

N_KEEP = 200  # simulated games kept per real game for distribution checks


def american_to_prob(ml: pd.Series) -> pd.Series:
    ml = ml.astype(float)
    return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))


def market_wp(games: pd.DataFrame) -> pd.Series:
    """De-vigged closing moneyline home win probability (NaN when missing)."""
    ph = american_to_prob(games["mkt_home_moneyline"])
    pa = american_to_prob(games["mkt_away_moneyline"])
    return pd.Series(ph / (ph + pa), index=games.index)


def load_inputs():
    games = read_processed("games")
    gf = read_processed("game_features")
    drives = read_processed("drives")
    pat = read_processed("pat")
    elo = compute_elo(games)
    gf = with_elo(gf, elo)
    gf = gf.merge(games[["game_id", "game_type"]].rename(columns={"game_type": "gt"}),
                  on="game_id")
    return games, gf, drives, pat


def _suffix(mode: str, qb_exits: bool = True, pass_shock: float = 0.0) -> str:
    return (("" if mode == "independent" else "_anchored") + ("" if qb_exits else "_noexits")
            + ("_shock" if pass_shock else ""))


def walk_forward(start_season: int = 2021, n_sims: int = 4000, seed: int = 0,
                 verbose: bool = True, end_season: int | None = None, save: bool = True,
                 mode: str = "independent", qb_exits: bool = True, pass_shock: float = 0.0,
                 **fit_kwargs):
    games, gf, drives, pat = load_inputs()
    test = games[(games["season"] >= start_season) & games["label_home_score"].notna()]
    if end_season is not None:
        test = test[test["season"] <= end_season]
    rng = np.random.default_rng(seed)
    preds, samples = [], []
    t0 = time.time()
    for (season, week), wk in test.groupby(["season", "week"], sort=True):
        cutoff = wk["gameday"].min()
        tm = fit_team_model(cutoff, drives, pat, gf, **fit_kwargs)
        # Home-team baseline from prior non-neutral games (previous 5 seasons).
        hist = games[(games["gameday"] < cutoff) & (games["season"] >= season - 5)
                     & games["label_home_score"].notna() & (games["neutral"] == 0)]
        hm = hist["label_home_score"] - hist["label_away_score"]
        base_wp = float(((hm > 0) + 0.5 * (hm == 0)).mean())
        base_margin = float(hm.mean())
        base_total = float((hist["label_home_score"] + hist["label_away_score"]).mean())
        for g in wk.itertuples():
            row = gf[gf["game_id"] == g.game_id].iloc[0]
            mu = replace(matchup_from_features(row), qb_exits=qb_exits, pass_shock=pass_shock)
            sseed = int(rng.integers(1 << 31))
            anc = {}
            if mode == "market_anchored":
                from .pipeline import odds_snapshot_for
                # backtest: a snapshot is usable only if taken before kickoff (enforced inside)
                line = get_line(pd.Series(g._asdict()),
                                snapshot=odds_snapshot_for(g.game_id, pd.Series(g._asdict()),
                                                           as_of=pd.Timestamp("2100-01-01", tz="UTC")))
                if line.complete:
                    a = solve_anchor(tm, mu, line.target_margin, line.target_total, n_sims, sseed)
                    r = a.sim
                    anc = {"anchored": True, "line_source": line.source, "anchor_margin": line.target_margin,
                           "anchor_total": line.target_total, "d_home": a.d_home, "d_away": a.d_away,
                           "converged": a.converged, "iterations": a.iterations,
                           "base_margin": a.base_margin, "base_total": a.base_total}
                else:
                    r = simulate(tm, mu, n_sims, seed=sseed)
                    anc = {"anchored": False, "fallback": line.note}
            else:
                r = simulate(tm, mu, n_sims, seed=sseed)
            preds.append({
                "game_id": g.game_id, "season": season, "week": week, "game_type": g.game_type,
                "gameday": g.gameday, "home_team": g.home_team, "away_team": g.away_team,
                "neutral": g.neutral,
                "sim_wp": r.home_win_prob(), "sim_margin": r.margin.mean(),
                "sim_total": r.total.mean(), "sim_home": r.home_score.mean(),
                "sim_away": r.away_score.mean(), "sim_drives": r.n_drives.mean(),
                "sim_ot": r.overtime.mean(),
                "elo_wp": row["elo_wp"],
                "base_wp": 0.5 if g.neutral else base_wp,
                "base_margin": 0.0 if g.neutral else base_margin, "base_total": base_total,
                "mkt_wp": np.nan, "mkt_spread": g.mkt_spread_line, "mkt_total": g.mkt_total_line,
                "home_score": g.label_home_score, "away_score": g.label_away_score,
                "overtime": g.label_overtime, "n_train_drives": tm.n_train_drives,
                "qb_source_home": row["home_qb_source"], "qb_source_away": row["away_qb_source"],
                "mode": mode, **anc,
                **{f"sim_qb_exit_{nm}": float((np.bincount(r.drives["sim"][(r.drives["side"] == sd) & r.drives["qb_out"]],
                                                           minlength=n_sims) > 0).mean())
                   for sd, nm in ((0, "home"), (1, "away"))},
            })
            k = min(N_KEEP, n_sims)
            samples.append(pd.DataFrame({
                "game_id": g.game_id, "home_score": r.home_score[:k],
                "away_score": r.away_score[:k], "n_drives": r.n_drives[:k],
                "overtime": r.overtime[:k]}))
        if verbose:
            print(f"{season} wk{week:>2}: {len(wk)} games, train drives={tm.n_train_drives}, "
                  f"{time.time() - t0:.0f}s")
    pr = pd.DataFrame(preds)
    gidx = games.set_index("game_id")
    pr["mkt_wp"] = market_wp(gidx.loc[pr["game_id"]]).to_numpy()
    sm = pd.concat(samples, ignore_index=True)
    if save:
        pr.to_parquet(PROCESSED / f"wf_predictions{_suffix(mode, qb_exits, pass_shock)}.parquet", index=False)
        sm.to_parquet(PROCESSED / f"wf_sim_samples{_suffix(mode, qb_exits, pass_shock)}.parquet", index=False)
    return pr, sm


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def anchor_check(pr: pd.DataFrame, tol: float = 0.25) -> dict:
    """Did every anchored game hit its targets? Also how far the market moved the model."""
    a = pr[pr["anchored"] == True]  # noqa: E712
    em = a["sim_margin"] - a["anchor_margin"]
    et = a["sim_total"] - a["anchor_total"]
    bad = a[(em.abs() > tol) | (et.abs() > tol)]
    dm = a["anchor_margin"] - a["base_margin"]
    dt = a["anchor_total"] - a["base_total"]
    return {"n_anchored": int(len(a)), "n_fallback": int((pr["anchored"] != True).sum()),  # noqa: E712
            "n_outside_tolerance": int(len(bad)), "games_outside_tolerance": bad["game_id"].tolist(),
            "max_abs_margin_err": float(em.abs().max()), "max_abs_total_err": float(et.abs().max()),
            "mean_solver_runs": float(a["iterations"].mean() + 1),
            "market_minus_model_margin_mean_abs": float(dm.abs().mean()),
            "market_minus_model_total_mean": float(dt.mean()),
            "market_minus_model_total_mean_abs": float(dt.abs().mean()),
            "score_logodds_shift_mean_abs": float(np.r_[a["d_home"], a["d_away"]].__abs__().mean()),
            "score_logodds_shift_p95_abs": float(np.quantile(np.abs(np.r_[a["d_home"], a["d_away"]]), .95)),
            "score_logodds_shift_max_abs": float(np.abs(np.r_[a["d_home"], a["d_away"]]).max())}


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def log_loss(p, y, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def evaluate(pr: pd.DataFrame) -> dict:
    pr = pr.copy()
    pr["margin"] = pr["home_score"] - pr["away_score"]
    pr["total"] = pr["home_score"] + pr["away_score"]
    nt = pr[pr["margin"] != 0]
    y = (nt["margin"] > 0).astype(float).to_numpy()
    has_mkt = nt["mkt_wp"].notna() & nt["mkt_spread"].notna() & nt["mkt_total"].notna()
    out = {"n_games": int(len(pr)), "n_ties_excluded_from_wp": int((pr["margin"] == 0).sum()),
           "n_games_with_market": int(has_mkt.sum()), "wp": {}, "score": {}}
    for name, col in [("model_sim", "sim_wp"), ("elo", "elo_wp"), ("home_baseline", "base_wp"),
                      ("market_close", "mkt_wp")]:
        m = has_mkt.to_numpy() if name == "market_close" else np.ones(len(nt), bool)
        p = nt[col].to_numpy()[m]
        out["wp"][name] = {"brier": brier(p, y[m]), "log_loss": log_loss(p, y[m]),
                           "accuracy": float(np.mean((p > 0.5) == (y[m] == 1))), "n": int(m.sum())}
    # Same subset for all, for an apples-to-apples comparison with market.
    m = has_mkt.to_numpy()
    out["wp_market_subset"] = {
        name: {"brier": brier(nt[col].to_numpy()[m], y[m]),
               "log_loss": log_loss(nt[col].to_numpy()[m], y[m])}
        for name, col in [("model_sim", "sim_wp"), ("elo", "elo_wp"),
                          ("home_baseline", "base_wp"), ("market_close", "mkt_wp")]}
    pm = pr.dropna(subset=["mkt_spread", "mkt_total"])
    for name, mcol, tcol, d in [("model_sim", "sim_margin", "sim_total", pr),
                                ("home_baseline", "base_margin", "base_total", pr),
                                ("model_sim_mkt_subset", "sim_margin", "sim_total", pm),
                                ("market_close", "mkt_spread", "mkt_total", pm)]:
        em = d["margin"] - d[mcol]
        et = d["total"] - d[tcol]
        out["score"][name] = {"margin_mae": float(em.abs().mean()),
                              "margin_rmse": float(np.sqrt((em ** 2).mean())),
                              "total_mae": float(et.abs().mean()),
                              "total_rmse": float(np.sqrt((et ** 2).mean())),
                              "total_bias": float(et.mean()), "n": int(len(d))}
    # Team score error (model mean team points vs actual)
    eh = np.concatenate([pr["home_score"] - pr["sim_home"], pr["away_score"] - pr["sim_away"]])
    out["score"]["model_sim"]["team_points_mae"] = float(np.abs(eh).mean())
    out["by_season"] = {}
    for s, d in nt.groupby("season"):
        yy = (d["margin"] > 0).astype(float).to_numpy()
        mk = d["mkt_wp"].notna().to_numpy()
        out["by_season"][int(s)] = {
            "n": int(len(d)),
            "model_brier": brier(d["sim_wp"].to_numpy(), yy),
            "model_logloss": log_loss(d["sim_wp"].to_numpy(), yy),
            "elo_brier": brier(d["elo_wp"].to_numpy(), yy),
            "base_brier": brier(d["base_wp"].to_numpy(), yy),
            "mkt_brier": brier(d["mkt_wp"].to_numpy()[mk], yy[mk]),
        }
    # Calibration of model WP by bucket
    b = pd.cut(nt["sim_wp"], [0, .2, .35, .5, .65, .8, 1.0], include_lowest=True)
    cal = nt.assign(y=y, b=b).groupby("b", observed=True).agg(
        n=("y", "size"), pred=("sim_wp", "mean"), actual=("y", "mean"))
    out["model_calibration"] = {str(k): {kk: float(vv) for kk, vv in v.items()}
                                for k, v in cal.iterrows()}
    return out


def league_validation(pr: pd.DataFrame, sm: pd.DataFrame, drives: pd.DataFrame) -> dict:
    """Simulated league-wide numbers vs what actually happened in the same games."""
    act = pr[["game_id", "home_score", "away_score", "overtime"]].copy()
    act_dr = drives[drives["game_id"].isin(act["game_id"])].groupby("game_id").size()
    act["n_drives"] = act["game_id"].map(act_dr)

    def summarize(df: pd.DataFrame) -> dict:
        m = df["home_score"] - df["away_score"]
        t = df["home_score"] + df["away_score"]
        am = m.abs()
        return {
            "points_per_team_game": float(np.r_[df["home_score"], df["away_score"]].mean()),
            "sd_team_points": float(np.r_[df["home_score"], df["away_score"]].std()),
            "total_mean": float(t.mean()), "total_sd": float(t.std()),
            "total_q10_50_90": [float(x) for x in t.quantile([.1, .5, .9])],
            "margin_sd": float(m.std()),
            "drives_per_game": float(df["n_drives"].mean()),
            "home_win_rate": float(((m > 0) + 0.5 * (m == 0)).mean()),
            "home_margin_mean": float(m.mean()),
            "pct_margin_eq_3": float((am == 3).mean()), "pct_margin_eq_7": float((am == 7).mean()),
            "pct_margin_le_7": float((am <= 7).mean()), "pct_margin_ge_17": float((am >= 17).mean()),
            "pct_ties": float((m == 0).mean()), "pct_overtime": float(df["overtime"].astype(float).mean()),
            "pct_team_shutout": float((np.r_[df["home_score"], df["away_score"]] == 0).mean()),
            "pct_team_40plus": float((np.r_[df["home_score"], df["away_score"]] >= 40).mean()),
        }

    out = {"simulated": summarize(sm), "actual": summarize(act.dropna(subset=["n_drives"]))}
    # Favorites: model favorite = sim_wp > .5 ; market favorite = mkt_wp > .5
    pr = pr.copy()
    margin = pr["home_score"] - pr["away_score"]
    nt = margin != 0
    fav_home = pr["sim_wp"] > 0.5
    fav_p = np.where(fav_home, pr["sim_wp"], 1 - pr["sim_wp"])
    fav_won = np.where(fav_home, margin > 0, margin < 0)
    out["favorites"] = {
        "model_favorite_predicted_win_rate": float(fav_p[nt].mean()),
        "model_favorite_actual_win_rate": float(fav_won[nt].mean()),
    }
    mk = pr["mkt_wp"].notna() & nt
    mf_home = pr["mkt_wp"] > 0.5
    mf_won = np.where(mf_home, margin > 0, margin < 0)
    mf_p = np.where(mf_home, pr["mkt_wp"], 1 - pr["mkt_wp"])
    out["favorites"]["market_favorite_implied_win_rate"] = float(mf_p[mk].mean())
    out["favorites"]["market_favorite_actual_win_rate"] = float(mf_won[mk].mean())
    # Within simulated games: how often does the pregame (model) favorite win?
    s = sm.merge(pr[["game_id", "sim_wp"]], on="game_id")
    sm_m = s["home_score"] - s["away_score"]
    s_fav_home = s["sim_wp"] > 0.5
    s_won = np.where(s_fav_home, sm_m > 0, sm_m < 0)
    out["favorites"]["model_favorite_win_rate_in_simulations"] = float(s_won[sm_m != 0].mean())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2021)
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--reuse", action="store_true", help="re-score saved predictions")
    ap.add_argument("--players", action="store_true",
                    help="M2 player distribution/correlation walk-forward instead")
    ap.add_argument("--player-seasons", type=int, nargs="*", default=[2024, 2025])
    ap.add_argument("--mode", default="independent", choices=["independent", "market_anchored"])
    ap.add_argument("--qb-exits", default="on", choices=["on", "off"],
                    help="simulate starting-QB early exits (default on)")
    ap.add_argument("--pass-shock", default="off", choices=["on", "off"],
                    help="shared per-game passing shock (Milestone 2.7; default off)")
    ap.add_argument("--pass-shock-sd", type=float, help="shock size when on (default PASS_SHOCK_SD)")
    ap.add_argument("--tag", default="", help="suffix for --players/--props output files")
    ap.add_argument("--props", action="store_true",
                    help="M3: props at self-set lines, synthetic SGPs and real-line ROI walk-forward")
    ap.add_argument("--prop-seasons", type=int, nargs="*", default=[2024, 2025, 2026])
    a = ap.parse_args(argv)
    qx = a.qb_exits == "on"
    from .simulator import PASS_SHOCK_SD
    shk = (a.pass_shock_sd if a.pass_shock_sd is not None else PASS_SHOCK_SD) if a.pass_shock == "on" else 0.0
    sx = _suffix(a.mode, qx, shk) + a.tag
    if a.props:
        if a.reuse:
            pp = pd.read_parquet(PROCESSED / f"prop_wf_props{sx}.parquet")
            ps_ = pd.read_parquet(PROCESSED / f"prop_wf_sgp{sx}.parquet")
            pr_ = pd.read_parquet(PROCESSED / f"prop_wf_real{sx}.parquet")
        else:
            pp, ps_, pr_ = prop_walk_forward(tuple(a.prop_seasons), a.sims if a.sims != 4000 else 2000,
                                             mode=a.mode, qb_exits=qx, pass_shock=shk, tag=a.tag)
        rep_ = {"mode": a.mode, "qb_exits": a.qb_exits, "pass_shock_sd": shk,
                "seasons": a.prop_seasons, **prop_report(pp, ps_, pr_)}
        (PROCESSED / f"prop_wf_report{sx}.json").write_text(json.dumps(rep_, indent=2, default=float))
        print(json.dumps({k: rep_[k] for k in ("props_overall", "sgp_overall")}, indent=1, default=float))
        return
    if a.players:
        if a.reuse:
            pr = pd.read_parquet(PROCESSED / f"player_wf{sx}.parquet")
            sm = pd.read_parquet(PROCESSED / f"player_wf_samples{sx}.parquet")
            gc = pd.read_pickle(PROCESSED / f"player_wf_gamecorr{sx}.pkl")
        else:
            pr, sm, gc = player_walk_forward(tuple(a.player_seasons), a.sims if a.sims != 4000 else 1000,
                                             mode=a.mode, qb_exits=qx, tag=a.tag, pass_shock=shk)
        pd.set_option("display.width", 250)
        dist = player_distribution_report(pr, sm)
        corr = correlation_report(gc)
        print(dist.round(2).to_string(index=False))
        print(corr.round(3).to_string(index=False))
        (PROCESSED / f"player_wf_report{sx}.json").write_text(json.dumps(
            {"distributions": dist.to_dict("records"), "correlations": corr.to_dict("records"),
             "played_rate_by_role": pr.groupby("role")["played"].mean().to_dict()},
            indent=2, default=float))
        return
    if a.reuse:
        pr = pd.read_parquet(PROCESSED / f"wf_predictions{sx}.parquet")
        sm = pd.read_parquet(PROCESSED / f"wf_sim_samples{sx}.parquet")
    else:
        pr, sm = walk_forward(a.start, a.sims, mode=a.mode, qb_exits=qx, pass_shock=shk)
    res = {"mode": a.mode, "qb_exits": a.qb_exits, "pass_shock_sd": shk, "evaluation": evaluate(pr),
           "league_validation": league_validation(pr, sm, read_processed("drives"))}
    if a.mode == "market_anchored":
        res["NOTE"] = ("Anchored to the closing market: win-probability and score metrics here only "
                       "check that anchoring works. They are NOT evidence of model skill.")
        res["anchor_check"] = anchor_check(pr)
    (PROCESSED / f"wf_report{sx}.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))


# --------------------------------------------------------------------------- #
# Milestone 2: player distributions and correlations (walk-forward)
# --------------------------------------------------------------------------- #
PSTATS = {"pass_att": "attempts", "completions": "completions", "pass_yards": "passing_yards",
          "pass_td": "passing_tds", "targets": "targets", "receptions": "receptions",
          "rec_yards": "receiving_yards", "rec_td": "receiving_tds", "rush_att": "carries",
          "rush_yards": "rushing_yards", "rush_td": "rushing_tds"}


def _roles_from_model(tp) -> dict:
    """Pregame roles from the model's own shares: QB1, RB1/RB2 by carry share,
    WR1-3 and TE1 by target share."""
    out = {}
    if tp.qb_idx >= 0:
        out[tp.qb_idx] = "QB1"
    for pos, n, share in (("WR", 3, tp.tgt_share), ("TE", 1, tp.tgt_share), ("RB", 2, tp.car_share)):
        idx = [i for i, p in enumerate(tp.position) if p == pos]
        idx = sorted(idx, key=lambda i: -share[i])[:n]
        for r, i in enumerate(idx, 1):
            out[i] = f"{pos}{r}"
    return out


def player_walk_forward(seasons=(2024, 2025), n_sims: int = 1000, keep: int = 100, seed: int = 0,
                        verbose: bool = True, mode: str = "independent", qb_exits: bool = True,
                        tag: str = "", pass_shock: float = 0.0):
    from .ingest import load
    from .pipeline import load_context, simulate_game
    ctx = load_context()
    games = ctx.games
    test = games[games["season"].isin(seasons) & games["label_home_score"].notna()]
    ps = load("player_stats", list(seasons)).to_pandas().set_index(["game_id", "player_id"])
    pg = ctx.player_games.set_index(["game_id", "player_id"])   # snaps>0 or usage => played
    rng = np.random.default_rng(seed)
    rows, samples, gamecorr = [], [], []
    t0 = time.time()
    for (season, week), wk in test.groupby(["season", "week"], sort=True):
        cutoff = wk["gameday"].min()
        tm = fit_team_model(cutoff, ctx.drives, ctx.pat, ctx.gf)
        for g in wk.itertuples():
            sg = simulate_game(ctx, g.game_id, n_sims, seed=int(rng.integers(1 << 31)),
                               cutoff=cutoff, team_model=tm, mode=mode, qb_exits=qb_exits,
                               pass_shock=pass_shock)
            margin = sg.sim.home_score - sg.sim.away_score
            act_margin = g.label_home_score - g.label_away_score
            for side, tp in sg.players.teams.items():
                sgn = 1 if side == 0 else -1
                roles = _roles_from_model(tp)
                st = sg.players.stats[side]
                rec = {"game_id": g.game_id, "team": tp.team, "season": season,
                       "sim_margin": sgn * margin[:keep], "act_margin": sgn * act_margin,
                       "sim_total": (sg.sim.home_score + sg.sim.away_score)[:keep],
                       "act_total": g.label_home_score + g.label_away_score,
                       # did the starting QB leave early in each kept simulation?
                       "sim_qb_exit": (np.bincount(sg.sim.drives["sim"][(sg.sim.drives["side"] == side)
                                                                        & sg.sim.drives["qb_out"]],
                                                   minlength=n_sims) > 0)[:keep],
                       # team passing totals (all QBs), for game-level variance checks
                       "team_pass_att": st["pass_att"].sum(1)[:keep],
                       "team_pass_yards": sg.players.team_plays[side]["pass_yards"][:keep],
                       "team_targets": sg.players.team_plays[side]["targets"][:keep],
                       "team_designed_runs": sg.players.team_plays[side]["designed_runs"][:keep],
                       "team_drives": np.bincount(sg.sim.drives["sim"][sg.sim.drives["side"] == side],
                                                  minlength=n_sims)[:keep]}
                for k, role in roles.items():
                    pid = tp.ids[k]
                    played = (g.game_id, pid) in pg.index
                    a = ps.loc[(g.game_id, pid)] if (g.game_id, pid) in ps.index else None
                    r = {"game_id": g.game_id, "season": season, "team": tp.team,
                         "player_id": pid, "role": role, "played": played}
                    for s, col in PSTATS.items():
                        x = st[s][:, k]
                        r["sim_mean_" + s] = x.mean()
                        r["act_" + s] = (float(a[col]) if a is not None and col in a else 0.0) if played else np.nan
                        # NaN where he sat out that simulation (actuals exist only if he played)
                        rec[f"{role}_{s}"] = np.where(sg.players.active[side][:keep, k],
                                                      x[:keep], np.nan)
                        rec[f"act_{role}_{s}"] = r["act_" + s]
                    rows.append(r)
                    if played:
                        # compare like with like: actuals exist only when he played, so keep
                        # only simulations in which he was active
                        act_k = sg.players.active[side][:keep, k]
                        samples.append(pd.DataFrame({"role": role,
                                                     **{s: st[s][:keep, k][act_k] for s in PSTATS}}))
                gamecorr.append(rec)
        if verbose:
            print(f"{season} wk{week:>2}: {len(wk)} games, {time.time() - t0:.0f}s", flush=True)
    pr = pd.DataFrame(rows)
    sm = pd.concat(samples, ignore_index=True)
    sx = _suffix(mode, qb_exits, pass_shock) + tag
    pr.to_parquet(PROCESSED / f"player_wf{sx}.parquet", index=False)
    sm.to_parquet(PROCESSED / f"player_wf_samples{sx}.parquet", index=False)
    pd.to_pickle(gamecorr, PROCESSED / f"player_wf_gamecorr{sx}.pkl")
    return pr, sm, gamecorr


def player_distribution_report(pr: pd.DataFrame, sm: pd.DataFrame) -> pd.DataFrame:
    """Pooled simulated vs actual distribution per role and stat (players who played)."""
    show = {"QB1": ["pass_att", "completions", "pass_yards", "pass_td", "rush_yards"],
            "RB1": ["rush_att", "rush_yards", "receptions", "rec_yards"],
            "RB2": ["rush_att", "rush_yards"],
            "WR1": ["targets", "receptions", "rec_yards"], "WR2": ["targets", "receptions", "rec_yards"],
            "WR3": ["receptions", "rec_yards"], "TE1": ["targets", "receptions", "rec_yards"]}
    out = []
    played = pr[pr["played"]]
    for role, stats in show.items():
        a = played[played["role"] == role]
        s = sm[sm["role"] == role]
        for st in stats:
            out.append({"role": role, "stat": st, "n_player_games": len(a),
                        "act_mean": a["act_" + st].mean(), "sim_mean": s[st].mean(),
                        "act_sd": a["act_" + st].std(), "sim_sd": s[st].std(),
                        "act_p10": a["act_" + st].quantile(.1), "sim_p10": s[st].quantile(.1),
                        "act_p90": a["act_" + st].quantile(.9), "sim_p90": s[st].quantile(.9)})
    return pd.DataFrame(out)


def correlation_report(gamecorr: list) -> pd.DataFrame:
    """Correlations pooled across team-games: simulated (all kept sims) vs actual
    (one observation per team-game). Only team-games where both players played."""
    pairs = [("QB1_pass_yards", "WR1_rec_yards"), ("QB1_pass_yards", "WR2_rec_yards"),
             ("QB1_pass_yards", "TE1_rec_yards"), ("WR1_rec_yards", "WR2_rec_yards"),
             ("QB1_pass_td", "WR1_rec_td"), ("RB1_rush_att", "margin"),
             ("QB1_pass_att", "margin"), ("QB1_pass_yards", "total"), ("RB1_rush_yards", "QB1_pass_yards")]
    out = []
    for a, b in pairs:
        sa, sb, xa, xb = [], [], [], []
        for rec in gamecorr:
            def get(k, act):
                if k in ("margin", "total"):
                    return rec[("act_" if act else "sim_") + k]
                return rec.get(("act_" if act else "") + k)
            va, vb = get(a, True), get(b, True)
            if va is None or vb is None or (np.isscalar(va) and np.isnan(va)) or \
                    (np.isscalar(vb) and np.isnan(vb)):
                continue
            sa.append(get(a, False)); sb.append(get(b, False)); xa.append(va); xb.append(vb)
        if len(xa) < 30:
            continue
        sa, sb = np.concatenate(sa), np.concatenate(sb)
        ok = ~np.isnan(sa) & ~np.isnan(sb)
        sa, sb = sa[ok], sb[ok]
        out.append({"pair": f"{a} ~ {b}", "n_team_games": len(xa),
                    "sim_corr_pooled": float(np.corrcoef(sa, sb)[0, 1]),
                    "actual_corr": float(np.corrcoef(xa, xb)[0, 1])})
    return pd.DataFrame(out)



# --------------------------------------------------------------------------- #
# Milestone 3: props at self-set lines, synthetic SGPs, real-line prop ROI
# --------------------------------------------------------------------------- #
# Official nflverse weekly columns per market. Anytime TD is graded the way books
# settle it (any non-passing TD: rushing, receiving, return/fumble recovery);
# the simulator credits no return TDs, so it is slightly low there (README).
PROP_ACTUAL = {"passing_yards": "passing_yards", "passing_tds": "passing_tds",
               "pass_attempts": "attempts", "completions": "completions",
               "rushing_yards": "rushing_yards", "rush_attempts": "carries",
               "receptions": "receptions", "receiving_yards": "receiving_yards"}
TD_COLS = ("rushing_tds", "receiving_tds", "special_teams_tds", "fumble_recovery_tds")
ROLE_MARKETS = {
    "QB1": ["passing_yards", "passing_tds", "pass_attempts", "completions", "rushing_yards"],
    "RB1": ["rushing_yards", "rush_attempts", "receptions", "receiving_yards", "anytime_td"],
    "RB2": ["rushing_yards", "rush_attempts", "receptions", "anytime_td"],
    "WR1": ["receptions", "receiving_yards", "anytime_td"],
    "WR2": ["receptions", "receiving_yards", "anytime_td"],
    "WR3": ["receptions", "receiving_yards", "anytime_td"],
    "TE1": ["receptions", "receiving_yards", "anytime_td"],
}
# Self-set fixed lines (no prices: nflverse has no prop lines). Same grid for every
# player in a role group, so model probabilities spread over the whole range.
FIXED_LINES = {
    "passing_yards": [174.5, 224.5, 274.5], "passing_tds": [0.5, 1.5, 2.5],
    "pass_attempts": [29.5, 34.5, 39.5], "completions": [19.5, 24.5],
    "rushing_yards": [14.5, 29.5, 49.5, 74.5, 99.5], "rush_attempts": [4.5, 9.5, 14.5, 19.5],
    "receptions": [1.5, 2.5, 4.5, 6.5], "receiving_yards": [19.5, 39.5, 59.5, 79.5, 99.5],
}
# Synthetic same-game parlays per team: (name, legs). Leg = (who, market, side);
# who is a model role, "team" (this team's moneyline) or "game" (closing total).
# Player legs use the "median" self-set line (model median rounded to x.5) or a
# fixed line (SGP_FIXED, by role and market).
SGP_TEMPLATES = [
    ("QB+WR1", [("QB1", "passing_yards", "over"), ("WR1", "receiving_yards", "over")]),
    ("QB+WR2", [("QB1", "passing_yards", "over"), ("WR2", "receiving_yards", "over")]),
    ("QB+TE1", [("QB1", "passing_yards", "over"), ("TE1", "receiving_yards", "over")]),
    ("QB+WR1+WR2", [("QB1", "passing_yards", "over"), ("WR1", "receiving_yards", "over"),
                    ("WR2", "receiving_yards", "over")]),
    ("WR1+WR2", [("WR1", "receiving_yards", "over"), ("WR2", "receiving_yards", "over")]),
    ("QB+total", [("QB1", "passing_yards", "over"), ("game", "total", "over")]),
    ("RB1+ML", [("RB1", "rushing_yards", "over"), ("team", "moneyline", "win")]),
    ("QBatt+ML", [("QB1", "pass_attempts", "over"), ("team", "moneyline", "win")]),
    ("RB1+QB", [("RB1", "rushing_yards", "over"), ("QB1", "passing_yards", "over")]),
    ("ML+total", [("team", "moneyline", "win"), ("game", "total", "over")]),
    ("QBtd+WR1td", [("QB1", "passing_tds", "over"), ("WR1", "anytime_td", "yes")]),
]
SGP_FIXED = {("QB1", "passing_yards"): 249.5, ("QB1", "pass_attempts"): 34.5,
             ("QB1", "passing_tds"): 1.5, ("WR1", "receiving_yards"): 64.5,
             ("WR2", "receiving_yards"): 44.5, ("TE1", "receiving_yards"): 34.5,
             ("RB1", "rushing_yards"): 59.5}
SIM_MARKET = {"passing_yards": "pass_yards", "passing_tds": "pass_td", "pass_attempts": "pass_att",
              "completions": "completions", "rushing_yards": "rush_yards", "rush_attempts": "rush_att",
              "receptions": "receptions", "receiving_yards": "rec_yards"}
MIN_ACTIVE = 100   # simulations a player must be active in for a probability


def _sim_values(st, k, market):
    if market == "anytime_td":
        return st["rush_td"][:, k] + st["rec_td"][:, k]
    return st[SIM_MARKET[market]][:, k]


def _act_value(a, market):
    """Official value; None when the player did not play."""
    if a is None:
        return None
    if market == "anytime_td":
        return float(sum(float(a[c]) for c in TD_COLS if c in a and pd.notna(a[c])))
    v = a.get(PROP_ACTUAL[market])
    return float(v) if pd.notna(v) else 0.0


def _median_line(x):
    return float(np.floor(np.median(x))) + 0.5


def load_real_props(game_ids) -> pd.DataFrame:
    """Saved pre-kickoff prop prices (odds feed) for these games: per game, book
    and market the latest snapshot before kickoff, matched players only."""
    path = PROCESSED / "prop_snapshots.parquet"
    if not path.exists():
        return pd.DataFrame()
    p = pd.read_parquet(path)
    p = p[p["game_id"].isin(set(game_ids)) & p["player_id"].notna()].copy()
    p["t"] = p["snapshot_ts"].fillna(p["pulled_at"])
    p = p[p["t"] < p["commence_time"]]          # never a post-kickoff price
    last = p.groupby(["game_id", "book", "market_key"])["t"].transform("max")
    return p[p["t"] == last].reset_index(drop=True)


def prop_walk_forward(seasons=(2024, 2025, 2026), n_sims: int = 2000, seed: int = 0,
                      mode: str = "independent", qb_exits: bool = True, pass_shock: float = 0.0,
                      verbose: bool = True, save: bool = True, tag: str = ""):
    """Weekly walk-forward for Milestone 3. Every week the team model is refit on
    data before the week's first game and player usage on games before it.

    Returns three tables:
      props: one row per (player, market, self-set line): P(over | active) from
             the simulation, official result (players who did not play are VOID
             and dropped, as books void them).
      sgp:   synthetic same-game parlays (SGP_TEMPLATES): joint P = K/N over the
             simulations where every leg's player is active and no leg pushes,
             the independence benchmark (product of the leg rates in the same
             simulations), and whether every leg actually won.
      real:  every saved pre-kickoff prop price (odds feed) for these games, with
             the model's P(win), P(push) for that side and the official result.
    """
    from .ingest import load
    from .pipeline import load_context, simulate_game
    ctx = load_context()
    games = ctx.games
    test = games[games["season"].isin(seasons) & games["label_home_score"].notna()]
    stats = load("player_stats", list(seasons)).to_pandas().set_index(["game_id", "player_id"])
    played_idx = ctx.player_games.set_index(["game_id", "player_id"]).index
    real_all = load_real_props(test["game_id"])
    real_by_game = {k: v for k, v in real_all.groupby("game_id")} if len(real_all) else {}
    rng = np.random.default_rng(seed)
    props, sgps, reals = [], [], []
    t0 = time.time()
    for (season, week), wk in test.groupby(["season", "week"], sort=True):
        cutoff = wk["gameday"].min()
        tm = fit_team_model(cutoff, ctx.drives, ctx.pat, ctx.gf)
        for g in wk.itertuples():
            sg = simulate_game(ctx, g.game_id, n_sims, seed=int(rng.integers(1 << 31)), cutoff=cutoff,
                               team_model=tm, mode=mode, qb_exits=qb_exits, pass_shock=pass_shock)
            hs, as_ = sg.sim.home_score, sg.sim.away_score
            act_h, act_a = g.label_home_score, g.label_away_score
            tline = g.mkt_total_line
            base = {"game_id": g.game_id, "season": season, "week": week}

            def actual_row(pid):
                if (g.game_id, pid) not in played_idx:
                    return None
                return stats.loc[(g.game_id, pid)] if (g.game_id, pid) in stats.index else {}

            for side, tp in sg.players.teams.items():
                st = sg.players.stats[side]
                act = sg.players.active[side]
                roles = _roles_from_model(tp)
                by_role = {r: k for k, r in roles.items()}
                won = (hs > as_) if side == 0 else (as_ > hs)
                tie = hs == as_
                act_won = (act_h > act_a) if side == 0 else (act_a > act_h)
                # -------- props at self-set lines
                for k, role in roles.items():
                    if role not in ROLE_MARKETS or act[:, k].sum() < MIN_ACTIVE:
                        continue
                    pid = tp.ids[k]
                    a = actual_row(pid)
                    for mkt in ROLE_MARKETS[role]:
                        x = _sim_values(st, k, mkt)[act[:, k]]
                        if mkt == "anytime_td":
                            lines = [("yes", 0.5)]
                        else:
                            lines = [("median", _median_line(x))] + [("fixed", l) for l in FIXED_LINES[mkt]]
                        av = _act_value(a, mkt)
                        for kind, line in lines:
                            props.append({**base, "team": tp.team, "player_id": pid, "role": role,
                                          "market": mkt, "line_type": kind, "line": line,
                                          "p_over": float((x > line).mean()),
                                          "sim_mean": float(x.mean()), "p_active": float(act[:, k].mean()),
                                          "played": av is not None,
                                          "actual": av, "y": None if av is None else float(av > line)})
                # -------- synthetic SGPs
                for name, legs in SGP_TEMPLATES:
                    for variant in ("median", "fixed"):
                        ok = np.ones(n_sims, bool)
                        win = np.ones(n_sims, bool)
                        leg_w, act_ok, act_hit, desc = [], True, True, []
                        for who, mkt, sd in legs:
                            if who == "game":
                                if pd.isna(tline):
                                    act_ok = False
                                    break
                                tot = hs + as_
                                ok &= tot != tline
                                w = tot > tline
                                at = act_h + act_a
                                act_ok &= at != tline
                                act_hit &= at > tline
                                desc.append(f"total o{tline:g}")
                            elif who == "team":
                                ok &= ~tie
                                w = won
                                act_ok &= act_h != act_a
                                act_hit &= bool(act_won)
                                desc.append(f"{tp.team} ML")
                            else:
                                k = by_role.get(who)
                                if k is None:
                                    act_ok = False
                                    break
                                xs = _sim_values(st, k, mkt)
                                if mkt == "anytime_td":
                                    line = 0.5
                                elif variant == "median":
                                    xa = xs[act[:, k]]
                                    if len(xa) < MIN_ACTIVE:
                                        act_ok = False
                                        break
                                    line = _median_line(xa)
                                else:
                                    line = SGP_FIXED[(who, mkt)]
                                ok &= act[:, k]
                                w = xs > line
                                av = _act_value(actual_row(tp.ids[k]), mkt)
                                if av is None:
                                    act_ok = False
                                    break
                                act_hit &= av > line
                                desc.append(f"{who} {mkt} o{line:g}")
                            win &= w
                            leg_w.append(w)
                        if not act_ok or ok.sum() < MIN_ACTIVE or len(leg_w) != len(legs):
                            continue
                        if variant == "fixed" and all(m == "anytime_td" or w_ in ("team", "game")
                                                      for w_, m, _ in legs):
                            continue   # no fixed-line leg: identical to the median variant
                        p_legs = [float(w[ok].mean()) for w in leg_w]
                        sgps.append({**base, "team": tp.team, "template": name, "variant": variant,
                                     "legs": " + ".join(desc), "n_legs": len(legs),
                                     "n_valid_sims": int(ok.sum()), "p_joint": float(win[ok].mean()),
                                     "p_indep": float(np.prod(p_legs)),
                                     "p_legs": p_legs, "y": float(act_hit)})
            # -------- real saved prices (odds feed)
            rp = real_by_game.get(g.game_id)
            if rp is not None:
                cache = {}
                for r in rp.itertuples():
                    key = (r.player_id, r.market, r.point, r.side)
                    if key not in cache:
                        cache[key] = None
                        for side, tp in sg.players.teams.items():
                            if r.player_id in tp.ids:
                                k = tp.ids.index(r.player_id)
                                act = sg.players.active[side][:, k]
                                if act.sum() >= MIN_ACTIVE:
                                    x = _sim_values(sg.players.stats[side], k, r.market)[act]
                                    if r.market == "anytime_td":
                                        pw, pp = float((x >= 1).mean()), 0.0
                                        if r.side == "no":
                                            pw = 1 - pw
                                    else:
                                        pp = float((x == r.point).mean())
                                        pw = float((x > r.point).mean() if r.side == "over"
                                                   else (x < r.point).mean())
                                    cache[key] = (pw, pp, float(act.mean()))
                    m = cache[key]
                    av = _act_value(actual_row(r.player_id), r.market)
                    if av is None:
                        res = "void"
                    elif r.market == "anytime_td":
                        res = "win" if (av >= 1) == (r.side == "yes") else "loss"
                    else:
                        res = ("push" if av == r.point else
                               "win" if (av > r.point) == (r.side == "over") else "loss")
                    reals.append({**base, "book": r.book, "player_id": r.player_id,
                                  "player_raw": r.player_raw, "market": r.market, "side": r.side,
                                  "point": r.point, "price": r.price, "fair_prob": r.fair_prob,
                                  "snapshot": r.t, "source": r.source,
                                  "p_win": None if m is None else m[0], "p_push": None if m is None else m[1],
                                  "p_active": None if m is None else m[2],
                                  "actual": av, "result": res})
        if verbose:
            print(f"{season} wk{week:>2}: {len(wk)} games, {time.time() - t0:.0f}s", flush=True)
    props, sgps, reals = pd.DataFrame(props), pd.DataFrame(sgps), pd.DataFrame(reals)
    if save:
        sx = _suffix(mode, qb_exits, pass_shock) + tag
        props.to_parquet(PROCESSED / f"prop_wf_props{sx}.parquet", index=False)
        sgps.to_parquet(PROCESSED / f"prop_wf_sgp{sx}.parquet", index=False)
        reals.to_parquet(PROCESSED / f"prop_wf_real{sx}.parquet", index=False)
    return props, sgps, reals


def _calib(p, y, bins=(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0)):
    b = pd.cut(p, bins, include_lowest=True)
    t = pd.DataFrame({"p": p, "y": y, "b": b}).groupby("b", observed=True).agg(
        n=("y", "size"), pred=("p", "mean"), actual=("y", "mean"))
    return [{"bucket": str(k), **{kk: float(vv) for kk, vv in v.items()}} for k, v in t.iterrows()]


def _scores(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return {"n": int(len(p)), "mean_pred": float(p.mean()), "actual_rate": float(y.mean()),
            "brier": brier(p, y), "log_loss": log_loss(p, y, eps=1e-3)}


def prop_report(props: pd.DataFrame, sgp: pd.DataFrame, real: pd.DataFrame | None = None) -> dict:
    """Calibration of props (self-set lines) and synthetic SGPs; real-line ROI."""
    out = {}
    d = props[props["played"]].copy()
    d["y"] = d["y"].astype(float)
    out["props_overall"] = _scores(d["p_over"], d["y"])
    # Reference only: Brier of the in-sample base rate for the same role/market/line type
    # (a skill-score yardstick, not a pregame forecast).
    clim = d.groupby(["role", "market", "line_type", "line"])["y"].transform("mean")
    out["props_overall"]["brier_insample_base_rate"] = brier(clim.to_numpy(), d["y"].to_numpy())
    out["props_by_market"] = {}
    for (mkt, kind), x in d.groupby(["market", "line_type"]):
        c = x.groupby(["role", "line"])["y"].transform("mean")
        out["props_by_market"][f"{mkt}|{kind}"] = {**_scores(x["p_over"], x["y"]),
                                                    "brier_insample_base_rate": brier(c.to_numpy(), x["y"].to_numpy()),
                                                    "calibration": _calib(x["p_over"], x["y"])}
    out["props_by_season"] = {int(s): _scores(x["p_over"], x["y"]) for s, x in d.groupby("season")}
    out["props_calibration"] = _calib(d["p_over"], d["y"])
    out["props_void_rate"] = float(1 - props["played"].mean())
    out["props_mean_p_active_when_void"] = float(props.loc[~props["played"], "p_active"].mean()) \
        if (~props["played"]).any() else None
    # SGPs
    s = sgp.copy()
    out["sgp_overall"] = {"joint": _scores(s["p_joint"], s["y"]), "independence": _scores(s["p_indep"], s["y"])}
    out["sgp_by_template"] = {}
    for (t, v), x in s.groupby(["template", "variant"], sort=False):
        n, k = len(x), x["y"].sum()
        lo, hi = _wilson(int(k), n)
        out["sgp_by_template"][f"{t}|{v}"] = {
            "n": n, "mean_p_joint": float(x["p_joint"].mean()), "mean_p_indep": float(x["p_indep"].mean()),
            "actual_rate": float(k / n), "actual_ci95": [lo, hi],
            "brier_joint": brier(x["p_joint"], x["y"]), "brier_indep": brier(x["p_indep"], x["y"]),
            "log_loss_joint": log_loss(x["p_joint"], x["y"], 1e-3),
            "log_loss_indep": log_loss(x["p_indep"], x["y"], 1e-3)}
    out["sgp_calibration_joint"] = _calib(s["p_joint"], s["y"])
    out["sgp_calibration_indep"] = _calib(s["p_indep"], s["y"])
    out["sgp_by_season"] = {int(k): {"joint": _scores(x["p_joint"], x["y"]),
                                     "independence": _scores(x["p_indep"], x["y"])}
                            for k, x in s.groupby("season")}
    out["sgp_note"] = ("Synthetic SGPs from the same games share players and game outcomes, so they are "
                       "not independent observations; per-template rates and calibration are noisy.")
    if real is not None and len(real):
        out["real_lines"] = real_line_report(real)
    return out


def _wilson(k, n, z=1.959964):
    from .parlay import wilson_ci
    return [float(v) for v in wilson_ci(k, n, z)]


def real_line_report(real: pd.DataFrame, ev_thresholds=(0.0, 0.05), seed: int = 0) -> dict:
    """Saved real prices: model vs market calibration and flat-stake ROI.

    Betting rule (fixed in advance, not tuned): for each (game, player, market,
    line, side), take the best price across books (or one book); bet 1 unit when
    the model's EV at that price is >= the threshold. Settled on official stats:
    void (player did not play) and push return the stake. 95% CI by bootstrap
    over games (bets in a game are not independent)."""
    from .odds import american_to_decimal
    r = real.dropna(subset=["p_win"]).copy()
    out = {"n_rows": int(len(real)), "n_rows_priced": int(len(r)),
           "n_games": int(real["game_id"].nunique()), "books": sorted(real["book"].unique().tolist()),
           "snapshots": "latest pre-kickoff snapshot per game and book (2026 backfill + live pulls)"}
    # Calibration on decided bets, model vs the book's de-vigged probability: two-sided
    # markets, over side only (the under is its mirror image)
    dec = r[r["result"].isin(["win", "loss"]) & (r["side"] == "over")].copy()
    dec["y"] = (dec["result"] == "win").astype(float)
    dec["p_model"] = dec["p_win"] / (1 - dec["p_push"]).clip(lower=1e-9)
    two = dec.dropna(subset=["fair_prob"])
    out["calibration_two_sided"] = {
        "n": int(len(two)), "n_games": int(two["game_id"].nunique()),
        "model": _scores(two["p_model"].clip(1e-3, 1 - 1e-3), two["y"]),
        "market_devigged": _scores(two["fair_prob"].clip(1e-3, 1 - 1e-3), two["y"]),
        "by_market": {m: {"n": int(len(x)), "brier_model": brier(x["p_model"], x["y"]),
                          "brier_market": brier(x["fair_prob"], x["y"])}
                      for m, x in two.groupby("market")}}
    r["dec"] = american_to_decimal(r["price"].to_numpy(float))
    r["ev"] = r["p_win"] * (r["dec"] - 1) - (1 - r["p_win"] - r["p_push"])
    r["ret"] = np.select([r["result"] == "win", r["result"].isin(["push", "void"])], [r["dec"], 1.0], 0.0)
    keys = ["game_id", "player_id", "market", "point", "side"]
    rng = np.random.default_rng(seed)
    out["roi"] = {}
    for scope in ("best_price_all_books", "fanatics"):
        x = r if scope == "best_price_all_books" else r[r["book"] == "fanatics"]
        if x.empty:
            continue
        x = x.sort_values("dec", ascending=False).drop_duplicates(keys)
        for th in ev_thresholds:
            b = x[x["ev"] >= th]
            res = {"n_bets": int(len(b)), "n_games": int(b["game_id"].nunique())}
            if len(b):
                prof = b["ret"] - 1
                res.update({"units_profit": float(prof.sum()), "roi": float(prof.mean()),
                            "mean_model_ev": float(b["ev"].mean()),
                            "win_rate_decided": float((b["result"] == "win").sum()
                                                      / max(b["result"].isin(["win", "loss"]).sum(), 1))})
                gp = prof.groupby(b["game_id"]).agg(["sum", "size"])
                bs = []
                for _ in range(2000):
                    i = rng.integers(0, len(gp), len(gp))
                    bs.append(gp["sum"].to_numpy()[i].sum() / gp["size"].to_numpy()[i].sum())
                res["roi_ci95_bootstrap_games"] = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
                res["by_market"] = {m: {"n": int(len(y)), "roi": float((y["ret"] - 1).mean())}
                                    for m, y in b.groupby("market")}
            out["roi"][f"{scope}|ev>={th:g}"] = res
    out["note"] = (f"Real prices exist only for {out['n_games']} games (2026 weeks 1-7 backfill plus live "
                   "pulls of completed games). ROI on this sample is very noisy; it is not evidence "
                   "of profitability.")
    return out


if __name__ == "__main__":
    main()
