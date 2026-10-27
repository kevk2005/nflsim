"""Milestone 3: market board, QB-receiver warning, actual grading, parlay log,
and the real-line ROI / calibration math. Small hand-made tables, no API calls."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nflsim import app_support as A
from nflsim.markets import LOSS, PUSH, VOID, WIN, Leg

KICK = pd.Timestamp("2026-10-25 17:00", tz="UTC")


def _lines():
    base = {"game_id": "G1", "commence_time": KICK, "pulled_at": KICK - pd.Timedelta(hours=1),
            "book": "fanatics", "home_team": "PHI", "away_team": "DAL"}
    rows = []
    for t, snap in (("early", KICK - pd.Timedelta(hours=5)), ("late", KICK - pd.Timedelta(minutes=10)),
                    ("post", KICK + pd.Timedelta(minutes=5))):
        price = {"early": -150, "late": -170, "post": -400}[t]
        rows += [{**base, "snapshot_ts": snap, "market": "h2h", "outcome": "Philadelphia Eagles",
                  "team": "PHI", "point": np.nan, "price": price},
                 {**base, "snapshot_ts": snap, "market": "spreads", "outcome": "Dallas Cowboys",
                  "team": "DAL", "point": 3.5, "price": -110},
                 {**base, "snapshot_ts": snap, "market": "totals", "outcome": "Over",
                  "team": None, "point": 44.5, "price": -105}]
    rows.append({**base, "book": "draftkings", "snapshot_ts": KICK - pd.Timedelta(minutes=10),
                 "market": "h2h", "outcome": "Philadelphia Eagles", "team": "PHI", "point": np.nan,
                 "price": -160})
    return pd.DataFrame(rows)


def _props():
    base = {"game_id": "G1", "commence_time": KICK, "pulled_at": KICK - pd.Timedelta(hours=1),
            "book": "fanatics", "snapshot_ts": KICK - pd.Timedelta(minutes=10)}
    return pd.DataFrame([
        {**base, "market": "passing_yards", "market_key": "player_pass_yds", "player_raw": "Jalen Hurts",
         "player_id": "QB-PHI", "side": "over", "point": 224.5, "price": -115},
        {**base, "market": "receiving_yards", "market_key": "player_reception_yds_alternate",
         "player_raw": "A.J. Brown", "player_id": "WR-PHI", "side": "over", "point": 99.5, "price": 250},
        {**base, "market": "anytime_td", "market_key": "player_anytime_td", "player_raw": "Eagles D/ST",
         "player_id": None, "side": "yes", "point": np.nan, "price": 1200},
        {**base, "market": "anytime_td", "market_key": "player_anytime_td", "player_raw": "Hollywood Brown",
         "player_id": None, "side": "yes", "point": np.nan, "price": 400},
        {**base, "market": None, "market_key": "player_reception_longest", "player_raw": "A.J. Brown",
         "player_id": "WR-PHI", "side": "over", "point": 24.5, "price": -110},
    ])


def test_market_board_latest_pre_kickoff_and_greying():
    b = A.market_board("G1", "fanatics", _lines(), _props(), "PHI", "DAL")
    gl = b[b["group"] == "Game lines"]
    assert len(gl) == 3
    # latest pre-kickoff snapshot only: never the post-kickoff price, never another book
    assert gl.loc[gl["market"] == "Moneyline", "price"].item() == -170
    assert b["snapshot_time"].max() < KICK
    assert list(b["order"]) == sorted(b["order"])                 # game lines first
    legs = dict(zip(b["label"], b["leg"]))
    assert legs["PHI ML"] == Leg("moneyline", "PHI")
    assert legs["DAL +3.5"] == Leg("spread", "DAL", 3.5)
    assert legs["Over 44.5"] == Leg("total", "over", 44.5)
    p = b[b["group"] != "Game lines"].set_index("label")
    assert p.loc["Jalen Hurts passing yards over 224.5", "priceable"]
    alt = p.loc["A.J. Brown receiving yards over 99.5"]
    assert alt["priceable"] and alt["alternate"]
    assert not p.loc["Eagles D/ST anytime td yes", "priceable"]
    assert "not a player" in p.loc["Eagles D/ST anytime td yes", "reason"]
    assert "not matched" in p.loc["Hollywood Brown anytime td yes", "reason"]
    longest = b[b["market"].str.contains("longest")]
    assert len(longest) == 1 and not longest["priceable"].item()
    assert longest["leg"].item() is None


def test_market_board_partial_pull_keeps_other_markets():
    """A later pull of one market updates that market and leaves the rest visible."""
    p = _props()
    late = p[p["market_key"] == "player_pass_yds"].assign(
        snapshot_ts=KICK - pd.Timedelta(minutes=2), point=230.5, price=-110)
    b = A.market_board("G1", "fanatics", _lines(), pd.concat([p, late]), "PHI", "DAL")
    pyd = b[b["market"] == "passing yards"]
    assert len(pyd) == 1 and pyd["point"].item() == 230.5          # newest price for that market
    assert (b["market"] == "receiving yards").any()                # other markets still shown
    assert b["snapshot_time"].nunique() >= 2


def test_market_board_all_books_best_price():
    b = A.market_board_all("G1", _lines(), _props(), "PHI", "DAL")
    ml = b[b["market"] == "Moneyline"]
    assert len(ml) == 1 and ml["price"].item() == -160 and ml["book"].item() == "draftkings"
    assert ml["n_books"].item() == 2 and "best of 2" in ml["label"].item()
    assert (b["leg"].notna() == b["priceable"]).all()
    assert b["snapshot_time"].max() < KICK                          # still never post-kickoff
    plain = A.market_board_all("G1", _lines(), _props(), "PHI", "DAL", show_book=False)
    assert plain["price"].tolist() == b["price"].tolist()          # same pooled prices ...
    assert not plain["label"].str.contains("draftkings|fanatics|best of").any()   # ... as one book


def test_market_board_as_of_and_empty():
    b = A.market_board("G1", "fanatics", _lines(), _props(), "PHI", "DAL",
                       as_of=KICK - pd.Timedelta(hours=2))
    assert b.loc[b["market"] == "Moneyline", "price"].item() == -150
    assert len(A.market_board("G1", "nobook", _lines(), _props(), "PHI", "DAL")) == 0


def test_alt_cost_plan():
    from nflsim.odds_feed import ALT_PROP_MARKETS, market_from_key
    assert len(ALT_PROP_MARKETS) == 8 and "anytime_td" not in ALT_PROP_MARKETS
    assert A.alt_cost_per_game() == 8
    assert market_from_key("player_reception_yds_alternate") == "receiving_yards"
    assert market_from_key("player_reception_longest") is None


def test_qb_receiver_pairs():
    teams = {0: SimpleNamespace(team="PHI", ids=["QB-PHI", "WR-PHI", "TE-PHI", "RB-PHI"],
                                position=["QB", "WR", "TE", "RB"]),
             1: SimpleNamespace(team="DAL", ids=["QB-DAL", "WR-DAL"], position=["QB", "WR"])}
    q = Leg("passing_yards", "over", 224.5, "QB-PHI")
    w = Leg("receiving_yards", "over", 60.5, "WR-PHI")
    t = Leg("anytime_td", "yes", None, "TE-PHI")
    rb = Leg("rushing_yards", "over", 50.5, "RB-PHI")
    other = Leg("receptions", "over", 4.5, "WR-DAL")
    assert A.qb_receiver_pairs([q, w], teams) == [(q, w)]
    assert len(A.qb_receiver_pairs([q, w, t], teams)) == 2
    assert A.qb_receiver_pairs([q, rb, other, Leg("moneyline", "PHI")], teams) == []


def test_grade_leg_actual():
    g = ("PHI", "DAL", 24, 20)
    assert A.grade_leg_actual(Leg("moneyline", "PHI"), *g, None, True) == WIN
    assert A.grade_leg_actual(Leg("spread", "DAL", 3.5), *g, None, True) == LOSS
    assert A.grade_leg_actual(Leg("spread", "DAL", 4.0), *g, None, True) == PUSH
    assert A.grade_leg_actual(Leg("total", "under", 44.5), *g, None, True) == WIN
    assert A.grade_leg_actual(Leg("total", "over", 44.0), *g, None, True) == PUSH
    s = {"passing_yards": 230, "receiving_tds": 0, "rushing_tds": 0, "special_teams_tds": 1}
    assert A.grade_leg_actual(Leg("passing_yards", "over", 224.5, "x"), *g, s, True) == WIN
    assert A.grade_leg_actual(Leg("passing_yards", "under", 224.5, "x"), *g, s, True) == LOSS
    assert A.grade_leg_actual(Leg("passing_yards", "over", 230, "x"), *g, s, True) == PUSH
    assert A.grade_leg_actual(Leg("anytime_td", "yes", None, "x"), *g, s, True) == WIN   # return TD counts
    assert A.grade_leg_actual(Leg("receptions", "over", 2.5, "x"), *g, s, False) == VOID


def test_parlay_log_roundtrip(tmp_path):
    p = tmp_path / "log.jsonl"
    legs = [Leg("moneyline", "PHI"), Leg("passing_yards", "over", 224.5, "QB-PHI", "Hurts o224.5")]
    A.log_parlay({"game_id": "G1", "legs": [A.leg_to_dict(l) for l in legs], "parlay_odds": 300,
                  "result": None}, p)
    e = A.load_log(p)
    assert len(e) == 1 and [A.leg_from_dict(d) for d in e[0]["legs"]] == legs


def test_real_line_report_math():
    from nflsim.backtest import real_line_report
    # Two games. Over at +100 with model 0.6 (EV +0.2) wins in G1 and loses in G2;
    # a -200 favorite with model 0.6 (EV -0.1) is never bet. Best price across books.
    rows = []
    for gid, res in (("G1", "win"), ("G2", "loss")):
        for book, price in (("fanatics", -110), ("draftkings", 100)):
            rows.append({"game_id": gid, "book": book, "player_id": "p", "player_raw": "P", "market": "receptions",
                         "side": "over", "point": 4.5, "price": price, "fair_prob": 0.5, "snapshot": KICK,
                         "source": "historical", "p_win": 0.6, "p_push": 0.0, "p_active": 1.0,
                         "actual": 5.0, "result": res})
        rows.append({"game_id": gid, "book": "fanatics", "player_id": "q", "player_raw": "Q",
                     "market": "receptions", "side": "over", "point": 2.5, "price": -200, "fair_prob": 0.64,
                     "snapshot": KICK, "source": "historical", "p_win": 0.6, "p_push": 0.0, "p_active": 1.0,
                     "actual": 3.0, "result": "win"})
    r = real_line_report(pd.DataFrame(rows))
    best = r["roi"]["best_price_all_books|ev>=0"]
    assert best["n_bets"] == 2 and best["units_profit"] == pytest.approx(0.0)   # +1 then -1 at +100
    fan = r["roi"]["fanatics|ev>=0"]
    assert fan["n_bets"] == 2 and fan["units_profit"] == pytest.approx(100 / 110 - 1)
    assert r["calibration_two_sided"]["n"] == 6


def test_prop_report_scores():
    from nflsim.backtest import prop_report
    props = pd.DataFrame({"game_id": ["a", "b", "c", "d"], "season": 2024, "week": 1, "team": "X",
                          "player_id": "p", "role": "WR1", "market": "receptions", "line_type": "fixed",
                          "line": 4.5, "p_over": [0.8, 0.2, 0.5, 0.5], "sim_mean": 4.0, "p_active": 0.9,
                          "played": [True, True, True, False], "actual": [6, 2, 5, None],
                          "y": [1.0, 0.0, 1.0, None]})
    sgp = pd.DataFrame({"game_id": ["a", "b"], "season": 2024, "week": 1, "team": "X", "template": "QB+WR1",
                        "variant": "median", "legs": "", "n_legs": 2, "n_valid_sims": 1000,
                        "p_joint": [0.3, 0.3], "p_indep": [0.25, 0.25], "p_legs": [[.5, .5], [.5, .5]],
                        "y": [1.0, 0.0]})
    r = prop_report(props, sgp)
    assert r["props_overall"]["n"] == 3                 # the void prop is dropped
    assert r["props_overall"]["brier"] == pytest.approx((0.04 + 0.04 + 0.25) / 3)
    assert r["props_void_rate"] == pytest.approx(0.25)
    t = r["sgp_by_template"]["QB+WR1|median"]
    assert t["n"] == 2 and t["actual_rate"] == 0.5
    assert t["brier_joint"] == pytest.approx(((0.3 - 1) ** 2 + 0.3 ** 2) / 2)
