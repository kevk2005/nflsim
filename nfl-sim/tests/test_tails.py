"""Milestone 4 game report: big-play and team-play stats from simulated plays,
threshold grid config, role baseline and shrinkage, market-line matching,
calibration bands, randomized PIT, keep test and trust labels, and the per-game
report build / export."""
import numpy as np
import pandas as pd
import pytest

from nflsim import tails as T

KICK = pd.Timestamp("2026-10-25 17:00", tz="UTC")


@pytest.fixture(scope="module")
def game():
    from nflsim.ingest import PROCESSED
    if not (PROCESSED / "drive_plays.parquet").exists():
        pytest.skip("run `python -m nflsim.players` first")
    from nflsim.pipeline import load_context, simulate_game
    return simulate_game(load_context(), "2024_05_BAL_CIN", 3000, seed=2)


def test_big_play_and_team_play_stats_consistent(game):
    for side in (0, 1):
        st, act = game.players.stats[side], game.players.active[side]
        assert (st["rec_long"] >= 0).all() and (st["rush_long"] >= 0).all()
        assert (st["rec_20"] <= st["receptions"]).all()
        assert ((st["rec_long"] >= 20) == (st["rec_20"] >= 1)).all()
        assert ((st["receptions"] == 0) <= (st["rec_long"] == 0)).all()          # no catch -> no long
        assert ((st["rush_att"] == 0) <= (st["rush_long"] == 0)).all()
        for s in ("rec_long", "rush_long", "rec_20"):
            assert (st[s][~act] == 0).all()                                      # inactive -> nothing
        assert (st["rec_long"] >= 40).any()                                      # real templates have big plays
        tp = game.players.team_plays[side]
        assert (tp["dropbacks"] + tp["designed_runs"] <= tp["plays"]).all()
        assert (tp["targets"] <= tp["dropbacks"]).all()
        q = game.players.teams[side].qb_idx
        assert (st["pass_att"][:, q] <= tp["dropbacks"]).all()                   # pass att excludes sacks/scrambles


def test_config_grid_and_levels(tmp_path):
    cfg = T.load_config()
    g = {s: [t.at_least for t in ts] for s, ts in cfg["by_stat"].items()}
    assert set(g) == set(T.STAT_COLS)
    assert g["rec_yards"] == list(range(5, 155, 5)) and g["pass_yards"] == list(range(150, 425, 25))
    assert g["receptions"] == list(range(1, 13))
    lv = {t.name: t.level for t in cfg["thresholds"]}
    assert (lv["rec_yards>=55"], lv["rec_yards>=60"], lv["rec_yards>=95"], lv["rec_yards>=100"]) == \
        ("low", "mid", "mid", "high")
    assert lv["tds>=1"] == "low" and lv["tds>=2"] == "high"
    t = next(t for t in cfg["thresholds"] if t.name == "rec_yards>=100")
    assert t.market == "receiving_yards" and t.market_line == 99.5 and "WR" in t.positions
    td = {t.at_least: t for t in cfg["by_stat"]["tds"]}
    assert td[1].market == "anytime_td" and td[1].market_line is None and td[2].market is None
    assert next(t for t in cfg["thresholds"] if t.name == "interceptions>=1").market is None
    bad = tmp_path / "bad.toml"
    bad.write_text(T.CONFIG.read_text().replace('stat = "rec_long"', 'stat = "longest_td"', 1))
    with pytest.raises(ValueError):
        T.load_config(bad)
    bad.write_text(T.CONFIG.read_text().replace("mid_from = 60", "mid_from = 200", 1))
    with pytest.raises(ValueError):
        T.load_config(bad)


def _base(window=3, k=2.0):
    ths = [T.Threshold("rec_yards", 100, "high", ("WR",))]
    games = [f"g{i}" for i in range(6)]
    pg = pd.DataFrame({"game_id": games, "player_id": "p", "team": "A", "season": 2024,
                       "gameday": pd.date_range("2024-09-08", periods=6, freq="7D")})
    pg = pd.concat([pg, pg.assign(player_id="q")])          # another WR1: never reaches 100
    act = pd.DataFrame({"game_id": games * 2, "player_id": ["p"] * 6 + ["q"] * 6,
                        **{c: 0.0 for c in T.STAT_COLS}})
    act.loc[act["player_id"] == "p", "rec_yards"] = [120, 10, 130, 140, 20, 0]
    roles = pg.assign(position="WR", rank=1)[["game_id", "player_id", "position", "rank"]]
    cfg = {"baseline": {"window_games": window, "prior_games": k, "prior_seasons": 2}}
    return T.Baseline(act, pg, roles, ths, cfg), pg


def test_baseline_role_rate_window_and_shrinkage():
    b, _ = _base()
    cutoff = pd.Timestamp("2024-10-13")                     # before g5: p's history g0..g4
    hist, n, role_r, role = b.get("p", "g5", "WR", cutoff, 2024)
    # role WR1 before cutoff: p 3/5 + q 0/5 = 0.3; last 3 games played: 130, 140, 20 -> 2 hits
    assert role == "WR1" and n == 3
    assert role_r["rec_yards>=100"] == pytest.approx(0.3)
    assert hist["rec_yards>=100"] == pytest.approx((2 + 2 * 0.3) / (3 + 2))
    assert list(b.history("p", cutoff)["rec_yards"]) == [130, 140, 20]
    _, n0, _, _ = b.get("p", "g0", "WR", pd.Timestamp("2024-09-08"), 2024)
    assert n0 == 0                                          # nothing from the game itself or later


def test_market_table_lines_and_lookup(tmp_path, monkeypatch):
    rows = []
    for book, over, under in (("dk", -120, 100), ("fd", -110, -110)):
        for side, price in (("over", over), ("under", under)):
            rows.append({"game_id": "G", "player_id": "p", "market": "receiving_yards",
                         "market_key": "player_reception_yds", "side": side, "point": 99.5, "price": price,
                         "fair_prob": None, "book": book, "snapshot_ts": KICK - pd.Timedelta(hours=1),
                         "pulled_at": KICK, "commence_time": KICK})
    rows.append({"game_id": "G", "player_id": "p", "market": "anytime_td", "market_key": "player_anytime_td",
                 "side": "yes", "point": np.nan, "price": 150, "fair_prob": None, "book": "dk",
                 "snapshot_ts": KICK - pd.Timedelta(hours=1), "pulled_at": KICK, "commence_time": KICK})
    rows.append({"game_id": "G", "player_id": "p", "market": "receiving_yards", "market_key": "player_reception_yds",
                 "side": "over", "point": 99.5, "price": 500, "fair_prob": 0.9, "book": "late",
                 "snapshot_ts": KICK + pd.Timedelta(minutes=5), "pulled_at": KICK, "commence_time": KICK})
    from nflsim.odds_feed import add_prop_devig
    df = pd.DataFrame(rows)
    df["pull_id"], df["player_raw"] = "x", "P"
    add_prop_devig(df).to_parquet(tmp_path / "prop_snapshots.parquet", index=False)
    monkeypatch.setattr(T, "PROCESSED", tmp_path)
    look = T._market_lookup(T.market_table(["G"]))
    r = look("G", "p", T.Threshold("rec_yards", 100, "high", ("WR",), "receiving_yards"))
    assert r is not None and not r.one_sided and r.n_books == 2       # post-kickoff book ignored
    from nflsim.odds import devig
    assert r.p_market == pytest.approx(np.median([devig(-120, 100)[0], devig(-110, -110)[0]]))
    assert look("G", "p", T.Threshold("rec_yards", 80, "mid", ("WR",), "receiving_yards")) is None
    td = look("G", "p", T.Threshold("tds", 1, "low", ("WR",), "anytime_td"))
    assert td.one_sided and td.p_market == pytest.approx(100 / 250)


def test_hit_probs_wilson_and_pit():
    x = np.array([0, 1, 1, 2, 5, 5, 5, 9])
    assert list(T.hit_probs(x, [0, 1, 2, 5, 6, 10])) == [1, 7 / 8, 5 / 8, 4 / 8, 1 / 8, 0]
    from nflsim.parlay import wilson_ci
    lo, hi = T.wilson(np.array([3, 50]), 200)
    assert (lo[0], hi[0]) == pytest.approx(wilson_ci(3, 200)) and (lo[1], hi[1]) == pytest.approx(wilson_ci(50, 200))
    assert T.pit(x, 5, 0.0) == pytest.approx(4 / 8) and T.pit(x, 5, 1.0) == pytest.approx(7 / 8)
    # a correct discrete forecast gives uniform randomized PIT (ties handled)
    rng = np.random.default_rng(0)
    sims = rng.poisson(2.0, 20000)
    p = np.array([T.pit(sims, y, u) for y, u in zip(rng.poisson(2.0, 4000), rng.random(4000))])
    assert abs((p < .1).mean() - .1) < .02 and abs((p < .5).mean() - .5) < .03 and abs((p > .9).mean() - .1) < .02


def _toy(n_games=60, per_game=600, bias=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows, srows = [], []
    for g in range(n_games):
        for i in range(per_game):
            p = rng.uniform(0, 1)
            lv = "low" if p > .3 else "mid" if p > .1 else "high"
            rows.append({"game_id": f"g{g}", "player_id": f"p{i}", "stat": "rec_yards", "at_least": 50,
                         "level": lv, "p_sim": float(np.clip(p + bias, 0, 1)), "p_role": p, "p_hist": p,
                         "played": True, "y": float(rng.random() < p)})
        for i in range(20):
            srows.append({"game_id": f"g{g}", "player_id": f"p{i}", "stat": "rec_yards", "played": True,
                          "pit_scored": i < 15, "pit": float(np.clip(rng.random() + bias, 0, 1))})
    td, sd = pd.DataFrame(rows), pd.DataFrame(srows)
    td["mode"], td["qb_exits"] = "market_anchored", False
    return td, sd


def test_calibration_bands():
    kr = T.load_config()["keep_rule"]
    d = pd.DataFrame({"game_id": ["a"] * 300 + ["b"] * 3, "p_sim": [0.05] * 300 + [1.0] * 3,
                      "y": [0.0] * 285 + [1.0] * 15 + [1.0] * 3})
    c = T.calibration(d, kr)
    assert [b["band"] for b in c["bands"]] == ["0–10%", "90–100%"]       # p = 1.0 in the top band
    assert c["bands"][0]["scored"] and c["bands"][0]["pass"] and c["bands"][1]["pass"] is None   # n < 200
    assert c["pass"] is True and c["n_scored_bands"] == 1
    d.loc[d.index[:300], "p_sim"] = 0.09                                  # sim 9% vs actual 5%
    assert T.calibration(d, kr)["pass"] is False


def test_keep_test_and_trust_labels():
    cfg = T.load_config()
    td, sd = _toy()
    rep = T.tail_report(td, sd, cfg)
    assert rep["keep_test"]["overall"] == "PASS"                          # calibrated by construction
    assert rep["trust"]["rec_yards"]["low"] == "trusted"
    assert rep["by_stat"]["rec_yards"]["percentile"]["n"] == 60 * 15      # scored rows only
    bad = T.tail_report(*_toy(bias=0.08), cfg)
    assert bad["keep_test"]["overall"] == "FAIL" and bad["keep_test"]["b_percentiles_pooled"] == "FAIL"
    lab = bad["trust"]["rec_yards"]["low"]
    assert lab.startswith("not trusted") and "calibration off" in lab and "percentiles off" in lab
    f = T.trust_labels(bad)
    assert f("rec_yards", "low") == lab and f("pass_yards", "high").startswith("not trusted")
    assert T.trust_labels(None)("rec_yards", "low") == "not trusted: no backtest"


def test_game_rows_and_report_build(game, tmp_path):
    from nflsim import gamereport as G
    from nflsim.pipeline import load_context
    cfg = T.load_config()
    ctx = load_context()
    base = T.Baseline(T.load_actuals(), ctx.player_games, ctx.roles, cfg["thresholds"], cfg)
    rep = G.build(ctx, base, cfg, "2024_05_BAL_CIN", n_sims=600, B=2, boot_sims=300, verbose=False,
                  names={}, now=pd.Timestamp("2024-10-06", tz="UTC"))
    thr, sts = rep["player_thresholds"], rep["player_stats"]
    assert {"QB", "RB", "WR", "TE"} <= set(sts["position"])
    assert not thr[thr["position"] == "WR"]["stat"].isin(["pass_yards", "rush_att"]).any()   # stats by position
    assert (thr["mc_lo"] <= thr["p_sim"]).all() and (thr["p_sim"] <= thr["mc_hi"]).all()
    mr = thr.dropna(subset=["model_sd"])
    assert len(mr) > 0.95 * len(thr)                                     # model range for (nearly) every row
    assert ((mr["model_lo"] <= mr["p_sim"] + 1e-12) & (mr["p_sim"] <= mr["model_hi"] + 1e-12)).all()
    assert np.allclose(thr["lift"], thr["p_sim"] - thr["p_role"])
    # hit % is non-increasing in the threshold, percentiles are ordered
    for _, g in thr.groupby(["player_id", "stat"]):
        assert (np.diff(g.sort_values("at_least")["p_sim"].to_numpy()) <= 1e-12).all()
    assert (sts["p10"] <= sts["p50"]).all() and (sts["p50"] <= sts["p90"]).all()
    tp = rep["team_probs"].set_index("outcome")["p"]
    assert tp["BAL win"] + tp["CIN win"] + tp["tie"] == pytest.approx(1)
    assert tp["BAL scores 20+"] >= tp["BAL scores 27+"] >= tp["BAL scores 34+"]
    assert len(rep["player_history"]) and len(rep["player_usage"])
    assert rep["player_history"].groupby("player_id").size().max() <= base.window
    hist_dates = pd.to_datetime(rep["player_history"]["gameday"])
    assert (hist_dates < pd.Timestamp("2024-10-06")).all()               # no peeking
    c = rep["correlations"]
    assert not c.loc[c["top"] != "", "kind"].ne("cross").any()           # top lists: cross pairs only
    assert (c.loc[c["top"] == "positive", "corr"] > 0).all() and (c.loc[c["top"] == "negative", "corr"] < 0).all()
    assert "does not use the opponent's defense" in rep["meta"]["opponent_note"]
    out = G.export(rep, tmp_path)
    back = G.load("2024_05_BAL_CIN", tmp_path)
    assert len(back["player_thresholds"]) == len(thr) and back["meta"]["n_sims"] == 600
    assert (back["correlations"]["top"] != "").sum() == (c["top"] != "").sum()
    assert "BAL @ CIN" in (out / "report.html").read_text() and G.saved_reports(tmp_path) == ["2024_05_BAL_CIN"]


def test_drivers_text():
    from nflsim import gamereport as G
    rows = [{"input": "target share", "value": 0.3, "reference": 0.2, "ratio": 1.5, "text": "target share 30% vs 20%"},
            {"input": "catch rate", "ratio": 1.02, "text": "catch rate"},
            {"input": "carry share", "value": 0.05, "reference": 0.10, "ratio": 0.5, "text": "carry share 5% vs 10%"},
            {"input": "P(active)", "ratio": np.nan, "text": "x"}]
    t = G.drivers_text(rows)
    assert t.startswith("carry share") and "target share" in t and "catch rate" not in t
    assert "close to the role" in G.drivers_text(rows[1:2])
    tiny = [{"input": "carry share", "value": 0.004, "reference": 0.012, "ratio": 0.33, "text": "carry share 0%"}]
    assert "close to the role" in G.drivers_text(tiny)                   # near-zero shares are not drivers
