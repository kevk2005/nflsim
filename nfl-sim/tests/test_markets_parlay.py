import numpy as np
import pytest

from nflsim import odds as O
from nflsim.markets import LOSS, PUSH, VOID, WIN, Leg, grade
from nflsim.parlay import evaluate, joint_from_grades, settle, wilson_ci


# --------------------------------------------------------------------------- odds
def test_conversions():
    assert np.isclose(O.american_to_decimal(-110), 1 + 100 / 110)
    assert np.isclose(O.american_to_decimal(150), 2.5)
    assert np.isclose(O.decimal_to_american(2.5), 150)
    assert np.isclose(O.decimal_to_american(1.5), -200)
    assert np.isclose(O.american_to_prob(-200), 2 / 3)
    for a in (-350, -110, 100, 120, 900):
        assert np.isclose(O.decimal_to_american(O.american_to_decimal(a)), a)
    with pytest.raises(ValueError):
        O.american_to_decimal(50)


def test_devig():
    pa, pb, over = O.devig(-110, -110)
    assert np.isclose(pa, 0.5) and np.isclose(pb, 0.5) and np.isclose(over, 2 / 1.909090909 - 1)
    pa, pb, _ = O.devig(-200, 170)
    assert np.isclose(pa + pb, 1) and pa > 0.6
    qa, qb, _ = O.devig(-200, 170, method="power")
    assert np.isclose(qa + qb, 1, atol=1e-6) and qa > pa   # power shades the longshot more


def test_fair_odds_and_ev():
    f = O.fair_odds(0.5)
    assert np.isclose(f["decimal"], 2) and np.isclose(f["american"], 100)
    assert np.isclose(O.fair_odds(0.25)["american"], 300)
    e = O.expected_value(0.55, O.american_to_decimal(-110))
    assert np.isclose(e["ev_per_unit"], 0.55 * (100 / 110) - 0.45)
    # a push returns the stake: 50% win at +100 with 10% push -> 0.5 - 0.4
    e = O.expected_value(0.5, 2.0, p_push=0.1)
    assert np.isclose(e["ev_per_unit"], 0.1) and np.isclose(e["p_loss"], 0.4)
    assert np.isclose(O.expected_value(0.5, 2.0)["ev_per_unit"], 0)
    assert np.isclose(O.expected_value_payouts(np.array([0, 2, 1, 2]))["ev_per_unit"], 0.25)


# --------------------------------------------------------------------------- grading
class _Sim:
    def __init__(self, h, a):
        self.home_score, self.away_score = np.array(h), np.array(a)


class _Players:
    def __init__(self, stats, active):
        self.stats, self.active = stats, active

    def player_stat(self, pid, stat):
        return None if pid not in self.stats else np.array(self.stats[pid].get(stat, [0] * len(self.active[pid])))

    def player_active(self, pid):
        return None if pid not in self.active else np.array(self.active[pid])


class _Game:
    def __init__(self, h, a, stats, active):
        self.n_sims = len(h)
        self.home_team, self.away_team = "KC", "BUF"
        self.sim = _Sim(h, a)
        self.players = _Players(stats, active)
        self.missing = []


def _game():
    stats = {"p1": {"rec_yards": [50, 60, 70, 80], "rec_td": [0, 1, 0, 0], "rush_td": [0, 0, 1, 0]},
             "p2": {"rec_yards": [10, 20, 30, 40]}}
    active = {"p1": [True, True, True, True], "p2": [True, False, True, True]}
    return _Game([24, 20, 17, 30], [21, 20, 24, 3], stats, active)


def test_grade_player_legs():
    g = _game()
    assert list(grade(Leg("receiving_yards", "over", 60, "p1"), g)) == [LOSS, PUSH, WIN, WIN]
    assert list(grade(Leg("receiving_yards", "under", 60, "p1"), g)) == [WIN, PUSH, LOSS, LOSS]
    assert list(grade(Leg("receiving_yards", "over", 15.5, "p2"), g)) == [LOSS, VOID, WIN, WIN]
    assert list(grade(Leg("anytime_td", "yes", None, "p1"), g)) == [LOSS, WIN, WIN, LOSS]
    assert list(grade(Leg("anytime_td", "no", None, "p1"), g)) == [WIN, LOSS, LOSS, WIN]
    assert (grade(Leg("receptions", "over", 2.5, "nobody"), g) == VOID).all()


def test_grade_game_legs():
    g = _game()   # margins KC: +3, 0, -7, +27 ; totals 45, 40, 41, 33
    assert list(grade(Leg("moneyline", "KC"), g)) == [WIN, PUSH, LOSS, WIN]
    assert list(grade(Leg("moneyline", "BUF"), g)) == [LOSS, PUSH, WIN, LOSS]
    assert list(grade(Leg("spread", "KC", -3), g)) == [PUSH, LOSS, LOSS, WIN]
    assert list(grade(Leg("spread", "BUF", 3.5), g)) == [WIN, WIN, WIN, LOSS]
    assert list(grade(Leg("total", "over", 41), g)) == [WIN, LOSS, PUSH, LOSS]
    with pytest.raises(ValueError):
        grade(Leg("moneyline", "NYJ"), g)
    with pytest.raises(ValueError):
        Leg("receptions", "over", None, "p1")


# --------------------------------------------------------------------------- parlay math
def test_k_over_n_and_ci():
    G = np.array([[WIN, WIN], [WIN, LOSS], [WIN, WIN], [LOSS, WIN], [WIN, PUSH]])
    j = joint_from_grades(G)
    assert j["K"] == 2 and j["N"] == 5 and np.isclose(j["p"], 0.4)
    lo, hi = j["ci95"]
    assert lo < 0.4 < hi
    lo, hi = wilson_ci(500, 1000)
    assert np.isclose((lo + hi) / 2, 0.5) and 0.468 < lo < 0.47


def test_settlement_rules():
    G = np.array([[WIN, WIN], [WIN, PUSH], [VOID, VOID], [LOSS, PUSH], [WIN, LOSS]])
    leg = np.array([2.0, 3.0])
    r = settle(G, "reduce", leg)
    assert list(r["payout"]) == [6.0, 2.0, 1.0, 0.0, 0.0]
    assert np.isclose(r["p_full_win"], 0.2) and np.isclose(r["p_reduced_win"], 0.2)
    assert np.isclose(r["p_refund"], 0.2) and np.isclose(r["p_loss"], 0.4)
    r = settle(G, "void_parlay", leg)
    assert list(r["payout"]) == [6.0, 1.0, 1.0, 0.0, 0.0]
    r = settle(G, "push_loses", leg)
    assert list(r["payout"]) == [6.0, 0.0, 0.0, 0.0, 0.0] and np.isclose(r["p_loss"], 0.8)
    r = settle(G, "reduce", leg, parlay_decimal=5.0)      # boosted/SGP price on full win
    assert r["payout"][0] == 5.0 and r["payout"][1] == 2.0
    r = settle(G, "reduce", None, parlay_decimal=5.0)     # reduced payout unknown without leg odds
    assert np.isnan(r["payout"][1]) and r["payout"][2] == 1.0


def test_evaluate_aggregation():
    g = _game()
    legs = [Leg("moneyline", "KC"), Leg("receiving_yards", "over", 55, "p1")]
    res = evaluate(g, legs, leg_odds=[-150, -110])
    # KC ML: W P L W ; p1 over 55: L W W W -> all win only in sim 4
    assert res.joint["K"] == 1 and res.joint["N"] == 4
    assert np.isclose(res.leg_rates[0]["p_win"], 0.5) and np.isclose(res.leg_rates[0]["p_push"], 0.25)
    assert np.isclose(res.independence_benchmark, 0.5 * 0.75)
    assert res.corr.shape == (2, 2) and np.isclose(res.corr[0, 0], 1)
    # payouts: sim1 L, sim2 reduced (ML push) -> 1.909, sim3 L, sim4 full win
    dec = O.american_to_decimal([-150, -110])
    expected = (0 + dec[1] + 0 + dec[0] * dec[1]) / 4 - 1
    assert np.isclose(res.ev["ev_per_unit"], expected)
