import numpy as np
import pandas as pd
import pytest

from nflsim.backtest import american_to_prob, brier, log_loss
from nflsim.simulator import simulate
from nflsim.team_model import OUTCOMES, STATES, STATIC_COLS, dynamic_features, state_index


@pytest.fixture(scope="module")
def sim(model, matchup):
    return simulate(model, matchup, n_sims=3000, seed=7, log_drives=True)


def test_scores_valid(sim):
    for s in (sim.home_score, sim.away_score):
        assert (s >= 0).all()
        assert (s != 1).all()  # 1 point is impossible for a team in a game
    assert np.array_equal(sim.home_q.sum(1), sim.home_score)
    assert np.array_equal(sim.away_q.sum(1), sim.away_score)
    # Overtime points only in overtime games
    assert (sim.home_q[~sim.overtime, 4] == 0).all() and (sim.away_q[~sim.overtime, 4] == 0).all()


def test_drive_log_matches_score(sim, matchup):
    log = sim.drive_log
    assert set(log["outcome"]) <= set(OUTCOMES)
    home = np.where(log["offense"] == matchup.home_team, log["off_pts"], log["def_pts"])
    away = np.where(log["offense"] == matchup.away_team, log["off_pts"], log["def_pts"])
    tot = pd.DataFrame({"sim": log["sim"], "h": home, "a": away}).groupby("sim").sum()
    assert np.array_equal(tot["h"].to_numpy(), sim.home_score)
    assert np.array_equal(tot["a"].to_numpy(), sim.away_score)
    # Points per drive outcome are consistent with football scoring
    td = log[log["outcome"] == "TD"]["off_pts"]
    assert td.isin([6, 7, 8]).all()
    assert (log[log["outcome"] == "FG"]["off_pts"] == 3).all()
    assert (log[log["outcome"] == "SAFETY"]["def_pts"] == 2).all()
    assert (log[~log["outcome"].isin(["TD", "FG"])]["off_pts"] == 0).all()


def test_possession_alternates(sim):
    log = sim.drive_log.sort_values(["sim", "drive"])
    nxt_off = log.groupby("sim")["offense"].shift(-1)
    nxt_half = log.groupby("sim")["half"].shift(-1)
    same_half = (nxt_half == log["half"]) & nxt_off.notna()
    keep = same_half & (log["outcome"] != "OPP_TD")
    assert (log.loc[keep, "offense"] != nxt_off[keep]).all()
    opp = same_half & (log["outcome"] == "OPP_TD")
    assert (log.loc[opp, "offense"] == nxt_off[opp]).all()


def test_clock(sim):
    log = sim.drive_log
    per_half = log[log["half"] <= 2].groupby(["sim", "half"])["seconds"].sum()
    assert (per_half <= 1800 + 1e-9).all()
    assert (per_half >= 1800 - 1e-9).all()  # each regulation half is played out
    assert (log["seconds"] >= 0).all()


def test_ties_only_after_ot(sim):
    tied = sim.home_score == sim.away_score
    assert sim.overtime[tied].all()


def test_playoff_no_ties(model, matchup):
    from dataclasses import replace
    r = simulate(model, replace(matchup, playoff=True), n_sims=3000, seed=3)
    assert (r.home_score != r.away_score).all()


def test_reproducible(model, matchup):
    a = simulate(model, matchup, 500, seed=11)
    b = simulate(model, matchup, 500, seed=11)
    assert np.array_equal(a.home_score, b.home_score)


def test_drive_probs_and_monotonic(model):
    base = np.zeros(len(STATIC_COLS))
    good = base.copy()
    good[STATIC_COLS.index("off_epa")] = 0.15
    rows = []
    for s in (base, good):
        X, _ = dynamic_features(s, STATIC_COLS, np.array([75.0]), np.array([1]),
                                np.array([900.0]), np.array([0.0]))
        rows.append(model.drive.predict_proba(X)[0])
    assert np.allclose([r.sum() for r in rows], 1)
    assert rows[1][OUTCOMES.index("TD")] > rows[0][OUTCOMES.index("TD")]


def test_late_game_state_effects(model, matchup):
    """Inside simulated games, trailing late passes more and plays faster than leading late."""
    log = simulate(model, matchup, n_sims=4000, seed=9, log_drives=True).drive_log
    log = log[log["plays"] >= 3]
    name = np.array(STATES)[log["state"].to_numpy()]
    for trail, lead in (("down4_8", "up4_8"), ("down9", "up9")):
        t = log[np.isin(name, [f"{trail}_q4", f"{trail}_5m"])]
        l_ = log[np.isin(name, [f"{lead}_q4", f"{lead}_5m"])]
        assert len(t) > 200 and len(l_) > 200
        assert t["pass_share"].mean() > l_["pass_share"].mean()
        assert t["seconds"].sum() / t["plays"].sum() < l_["seconds"].sum() / l_["plays"].sum()


def test_state_index():
    s = state_index(np.array([1, 1, 2, 2, 2, 3, 2]), np.array([900, 100, 1500, 600, 60, 400, 200]),
                    np.array([0, 0, -10, -3, 10, 0, -6]))
    names = [STATES[i] for i in s]
    assert names == ["normal", "h1_2m", "normal", "down1_3_q4", "up9_2m", "tied_q4", "down4_8_5m"]


def test_metrics():
    assert brier(np.array([1.0, 0.0]), np.array([1, 0])) == 0
    assert abs(log_loss(np.array([0.5]), np.array([1])) - np.log(2)) < 1e-9
    p = american_to_prob(pd.Series([-150, 130]))
    assert np.allclose(p, [0.6, 100 / 230])


def test_templates_and_tables_only_from_past(data):
    """Play templates, pace and field-position tables come only from drives
    played before the cutoff (no leakage of the game being simulated)."""
    from nflsim.simulator import matchup_from_features
    from nflsim.team_model import FieldPosTable, fit_team_model
    drives = data["drives"]
    date_of = drives.set_index("drive_uid")["game_date"]
    for gid in ("2023_01_DET_KC", "2024_10_DET_HOU", "2025_05_TB_SEA"):
        row = data["gf"][data["gf"].game_id == gid].iloc[0]
        cutoff = row["gameday"]
        tm = fit_team_model(cutoff, drives, data["pat"], data["gf"])
        assert (date_of.loc[tm.pace.uid] < cutoff).all()
        # the game's own drives are not available as templates
        own = set(drives.loc[drives.game_id == gid, "drive_uid"])
        assert own and not own & set(tm.pace.uid)
        # every template actually drawn in a simulation is from before the cutoff
        r = simulate(tm, matchup_from_features(row), 2000, seed=0)
        assert (date_of.loc[np.unique(r.drives["uid"])] < cutoff).all()
        # field-position table equals one rebuilt from strictly-earlier drives only
        past = drives[(drives.game_date < cutoff) & (drives.season > drives[drives.game_date < cutoff].season.max() - 3)
                      & (drives.half <= 2)]
        assert np.array_equal(FieldPosTable.fit(past).kickoff, tm.fieldpos.kickoff)
