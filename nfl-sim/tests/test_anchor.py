import numpy as np
import pandas as pd
import pytest

from nflsim.anchor import TOL, get_line

N = 3000


@pytest.fixture(scope="module")
def ctx():
    from nflsim.ingest import PROCESSED
    if not (PROCESSED / "drive_plays.parquet").exists():
        pytest.skip("run `python -m nflsim.players` first")
    from nflsim.pipeline import load_context
    return load_context()


def _pick(ctx):
    g = ctx.games.dropna(subset=["mkt_spread_line", "mkt_total_line"])
    g = g[g["season"] >= 2018]
    return {
        "big_home_fav": g.loc[g["mkt_spread_line"].idxmax(), "game_id"],
        "big_away_fav": g.loc[g["mkt_spread_line"].idxmin(), "game_id"],
        "high_total": g.loc[g["mkt_total_line"].idxmax(), "game_id"],
        "low_total": g.loc[g["mkt_total_line"].idxmin(), "game_id"],
        "typical": "2024_01_BAL_KC",
    }


def test_anchor_hits_targets(ctx):
    from nflsim.pipeline import simulate_game
    for name, gid in _pick(ctx).items():
        g = simulate_game(ctx, gid, N, seed=5, mode="market_anchored")
        a = g.anchor
        assert a["mode"] == "market_anchored", name
        assert a["converged"], (name, a)
        assert abs(g.sim.margin.mean() - a["target_margin"]) <= TOL, (name, a)
        assert abs(g.sim.total.mean() - a["target_total"]) <= TOL, (name, a)


def test_sign_convention(ctx):
    from nflsim.pipeline import simulate_game
    gid = _pick(ctx)["big_home_fav"]
    row = ctx.games[ctx.games.game_id == gid].iloc[0]
    assert row["mkt_spread_line"] > 10            # nflverse: positive = home favored
    assert row["mkt_home_moneyline"] < 0          # ...and the home team is the ML favorite
    g = simulate_game(ctx, gid, N, seed=1, mode="market_anchored")
    assert g.anchor["target_margin"] == row["mkt_spread_line"]
    assert g.sim.margin.mean() > 10
    # user-supplied sportsbook spread: home -7 means home expected to win by 7
    g = simulate_game(ctx, "2024_01_BAL_KC", N, seed=1, mode="market_anchored",
                      home_spread=-7.0, total=50.0)
    assert g.anchor["target_margin"] == 7.0 and "user" in g.anchor["source"]
    assert abs(g.sim.margin.mean() - 7.0) <= TOL and abs(g.sim.total.mean() - 50.0) <= TOL
    g = simulate_game(ctx, "2024_01_BAL_KC", N, seed=1, mode="market_anchored",
                      home_spread=6.5, total=40.0)   # home +6.5 underdog
    assert g.sim.margin.mean() < -6


def test_fallback_without_line(ctx):
    from nflsim.pipeline import simulate_game
    assert not get_line(pd.Series({"season": 2026, "mkt_spread_line": np.nan,
                                   "mkt_total_line": np.nan})).complete
    assert not get_line(pd.Series({"season": 2026, "mkt_spread_line": 3.0,
                                   "mkt_total_line": None})).complete
    no_line = ctx.games[ctx.games["mkt_spread_line"].isna() | ctx.games["mkt_total_line"].isna()]
    no_line = no_line[no_line["game_id"].isin(ctx.gf["game_id"])]
    if no_line.empty:
        pytest.skip("every scheduled game currently has a line")
    gid = no_line["game_id"].iloc[0]
    g = simulate_game(ctx, gid, 500, seed=0, mode="market_anchored")
    assert g.anchor["mode"] == "independent" and "fallback" in g.anchor
    assert "FELL BACK" in g.missing[0]
    ind = simulate_game(ctx, gid, 500, seed=0, mode="independent")
    assert np.array_equal(g.sim.home_score, ind.sim.home_score)   # exactly independent mode
    # a user-supplied line makes anchoring possible again
    g = simulate_game(ctx, gid, N, seed=0, mode="market_anchored", home_spread=-3.0, total=44.0)
    assert g.anchor["mode"] == "market_anchored"
    assert abs(g.sim.margin.mean() - 3.0) <= TOL


def test_only_offense_changes(ctx):
    """Anchoring must not touch player shares or anything except the offense terms."""
    from nflsim.pipeline import simulate_game
    a = simulate_game(ctx, "2024_01_BAL_KC", 500, seed=2, mode="independent")
    b = simulate_game(ctx, "2024_01_BAL_KC", 500, seed=2, mode="market_anchored")
    for side in (0, 1):
        ta, tb = a.players.teams[side], b.players.teams[side]
        assert ta.ids == tb.ids
        assert np.array_equal(ta.tgt_share, tb.tgt_share) and np.array_equal(ta.car_share, tb.car_share)
    from nflsim.anchor import shifted
    from nflsim.simulator import matchup_from_features
    mu = matchup_from_features(ctx.gf[ctx.gf.game_id == "2024_01_BAL_KC"].iloc[0])
    m2 = shifted(mu, 0.3, -0.2)
    assert np.array_equal(m2.home_static, mu.home_static) and np.array_equal(m2.away_static, mu.away_static)
    assert m2.score_shift == (0.3, -0.2) and mu.score_shift == (0.0, 0.0)


def test_shift_is_monotone(ctx, ):
    from nflsim.anchor import shifted
    from nflsim.simulator import matchup_from_features, simulate
    from nflsim.team_model import fit_team_model
    row = ctx.gf[ctx.gf.game_id == "2024_01_BAL_KC"].iloc[0]
    tm = fit_team_model(row["gameday"], ctx.drives, ctx.pat, ctx.gf)
    mu = matchup_from_features(row)
    pts = [simulate(tm, shifted(mu, s, 0.0), 2000, seed=3).home_score.mean() for s in (-1, -0.5, 0, 0.5, 1)]
    assert all(b > a for a, b in zip(pts, pts[1:]))


@pytest.mark.parametrize("mode", ["independent", "market_anchored"])
def test_qb_exit_switch(ctx, mode):
    from nflsim.pipeline import simulate_game
    on = simulate_game(ctx, "2024_05_BAL_CIN", N, seed=4, mode=mode)
    off = simulate_game(ctx, "2024_05_BAL_CIN", N, seed=4, mode=mode, qb_exits=False)
    assert on.sim.drives["qb_out"].any()
    assert not off.sim.drives["qb_out"].any()
    for side in (0, 1):
        b = off.players.teams[side].qb2_idx
        if b >= 0:
            assert (off.players.stats[side]["pass_att"][:, b] == 0).all()
    assert any("QB early exits: OFF" in m for m in off.missing)
    if mode == "market_anchored":
        for g in (on, off):
            assert abs(g.sim.margin.mean() - g.anchor["target_margin"]) <= TOL
            assert abs(g.sim.total.mean() - g.anchor["target_total"]) <= TOL
