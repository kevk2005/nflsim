"""No-future-data tests: features for a game must not change when later results change."""
import numpy as np
import pandas as pd
import polars as pl

from nflsim.features import compute_qb_features, compute_team_ratings
from nflsim.ingest import normalize_teams
from nflsim.team_model import compute_elo


def test_team_normalization():
    df = pl.DataFrame({"home_team": ["OAK", "SD", "STL", "KC"], "away_team": ["LAR", "JAC", "LV", "LA"]})
    out = normalize_teams(df, "schedules")
    assert out["home_team"].to_list() == ["LV", "LAC", "LA", "KC"]
    assert out["away_team"].to_list() == ["LA", "JAX", "LV", "LA"]


def test_asof_date_before_game(data):
    gf = data["gf"]
    has = gf["asof_date"].notna()
    assert (gf.loc[has, "asof_date"] < gf.loc[has, "gameday"]).all()
    assert (gf["weather"] == "unavailable").all()


def _scramble_after(games, cutoff):
    """Copy of games where every result on/after cutoff is replaced by garbage."""
    g = games.copy()
    late = g["gameday"] >= cutoff
    rng = np.random.default_rng(0)
    g.loc[late & g["label_home_score"].notna(), "label_home_score"] = rng.integers(0, 60, (late & g["label_home_score"].notna()).sum())
    return g, late


def test_team_ratings_ignore_future(data):
    games = data["games"][data["games"]["season"].between(2021, 2022)]
    tg = data["team_games"][data["team_games"]["game_id"].isin(games["game_id"])]
    cutoff = pd.Timestamp("2022-10-01")
    base = compute_team_ratings(games, tg)
    # Corrupt all play-level data on/after cutoff.
    late_ids = set(games.loc[games["gameday"] >= cutoff, "game_id"])
    tg2 = tg.copy()
    m = tg2["game_id"].isin(late_ids)
    tg2.loc[m, "epa"] = 99.0
    tg2.loc[m, "succ"] = 0.0
    alt = compute_team_ratings(games, tg2)
    pre_ids = set(games.loc[games["gameday"] <= cutoff, "game_id"])
    a = base[base["game_id"].isin(pre_ids)].reset_index(drop=True)
    b = alt[alt["game_id"].isin(pre_ids)].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    # And the corruption does change later games (sanity check of the test itself).
    assert not np.allclose(base.loc[~base["game_id"].isin(pre_ids), "off_epa"],
                           alt.loc[~alt["game_id"].isin(pre_ids), "off_epa"])


def test_elo_ignores_future(data):
    games = data["games"][data["games"]["season"].between(2020, 2022)]
    cutoff = pd.Timestamp("2022-10-01")
    g2, late = _scramble_after(games, cutoff)
    a = compute_elo(games).set_index("game_id")
    b = compute_elo(g2).set_index("game_id")
    pre = games.loc[games["gameday"] <= cutoff, "game_id"]
    pd.testing.assert_frame_equal(a.loc[pre], b.loc[pre])


def test_qb_features_ignore_future(data):
    games = data["games"][data["games"]["season"].between(2022, 2023)]
    tg = data["team_games"][data["team_games"]["game_id"].isin(games["game_id"])]
    qbg = data["qb_games"][data["qb_games"]["game_id"].isin(games["game_id"])]
    ratings = compute_team_ratings(games, tg)
    cutoff = pd.Timestamp("2023-10-01")
    late_ids = set(games.loc[games["gameday"] >= cutoff, "game_id"])
    q2 = qbg.copy()
    q2.loc[q2["game_id"].isin(late_ids), "qb_epa"] = -50.0
    a = compute_qb_features(games, qbg, ratings)
    b = compute_qb_features(games, q2, ratings)
    pre = set(games.loc[games["gameday"] <= cutoff, "game_id"])
    pd.testing.assert_frame_equal(a[a.game_id.isin(pre)].reset_index(drop=True),
                                  b[b.game_id.isin(pre)].reset_index(drop=True))


def test_drive_table_sane(data):
    d = data["drives"]
    assert d["yardline_100"].between(1, 99).all()
    assert d["half_seconds_remaining"].between(0, 1800).all()
    per_game = d.groupby("game_id").size()
    assert 18 < per_game.mean() < 26


def test_qb_delta_uses_rating_mix_not_last_game():
    """A backup on his 3rd start must still be compared with the injured starter
    whose games dominate the team rating (regression test for the M1 QB bug)."""
    n = 13
    days = pd.date_range("2016-09-11", periods=n, freq="7D")
    games = pd.DataFrame({"game_id": [f"g{i}" for i in range(n)], "season": 2016,
                          "week": range(1, n + 1), "gameday": days,
                          "home_team": "ZZA", "away_team": "ZZB"})
    # ZZA: starter S plays games 0-9 (good), backup B plays games 10-11 (bad).
    qbg = pd.DataFrame({"game_id": [f"g{i}" for i in range(12)], "team": "ZZA",
                        "qb_id": ["S"] * 10 + ["B"] * 2, "dropbacks": 35,
                        "qb_epa": [0.25 * 35] * 10 + [-0.25 * 35] * 2})
    ratings = pd.DataFrame({"game_id": games["game_id"], "lg_epa": 0.0})
    out = compute_qb_features(games, qbg, ratings).set_index(["game_id", "team"])
    last = out.loc[("g12", "ZZA")]
    assert last["qb_id"] == "B" and not last["qb_changed"]
    assert last["qb_mix_share"] < 0.5
    assert last["qb_delta"] < -0.1   # old definition gave exactly 0 here
