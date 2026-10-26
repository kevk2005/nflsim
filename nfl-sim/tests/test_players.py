"""Rule 3 consistency on thousands of simulated games (with players)."""
import numpy as np
import pytest

from nflsim.team_model import OUT_IDX

GAMES = ["2023_01_DET_KC", "2024_05_BAL_CIN", "2025_10_DET_WAS"]
N = 3000  # x 3 games = 9,000 simulated games


@pytest.fixture(scope="module")
def ctx():
    from nflsim.ingest import PROCESSED
    if not (PROCESSED / "drive_plays.parquet").exists():
        pytest.skip("run `python -m nflsim.players` first")
    from nflsim.pipeline import load_context
    return load_context()


@pytest.fixture(scope="module", params=["independent", "market_anchored"])
def sims(ctx, request):
    from nflsim.pipeline import simulate_game
    out = [simulate_game(ctx, g, N, seed=i, mode=request.param) for i, g in enumerate(GAMES)]
    if request.param == "market_anchored":
        assert all(g.anchor["mode"] == "market_anchored" for g in out)
    return out


def test_player_consistency(sims):
    for g in sims:
        for side in (0, 1):
            st = g.players.stats[side]
            tp = g.players.teams[side]
            team = g.players.team_plays[side]
            q, b = tp.qb_idx, tp.qb2_idx
            assert q >= 0
            qbs = [q] + ([b] if b >= 0 else [])
            assert (st["receptions"] <= st["targets"]).all()
            assert np.array_equal(st["receptions"].sum(1), team["completions"])
            assert np.array_equal(st["completions"][:, qbs].sum(1), team["completions"])
            assert np.array_equal(st["rec_yards"].sum(1), st["pass_yards"][:, qbs].sum(1))
            assert np.array_equal(st["pass_yards"][:, qbs].sum(1), team["pass_yards"])
            assert np.array_equal(st["rec_td"].sum(1), st["pass_td"][:, qbs].sum(1))
            assert (st["completions"] <= st["pass_att"]).all()
            # only the starter and (after a simulated exit) the backup throw
            others = np.delete(np.arange(len(tp.ids)), qbs)
            assert (st["pass_att"][:, others] == 0).all()
            D = g.sim.drives
            exited = np.bincount(D["sim"][(D["side"] == side) & D["qb_out"]], minlength=g.n_sims) > 0
            if b >= 0:
                assert (st["pass_att"][~exited, b] == 0).all() and (st["rush_att"][~exited, b] == 0).all()
                assert g.players.active[side][exited, b].all()
            # the starter is never voided by an exit: he played, so his props grade
            assert g.players.active[side][:, q].all()
            # offensive TD plays == TD drives, except kick/punt-return TD templates
            off_td = st["rec_td"].sum(1) + st["rush_td"].sum(1)
            assert np.array_equal(off_td, team["off_td_plays"])
            assert (off_td <= team["td_drives"]).all()
            assert off_td.sum() >= 0.95 * team["td_drives"].sum()
            # inactive players record nothing
            act = g.players.active[side]
            for s in st.values():
                assert (s[~act] == 0).all()


def test_points_match_scoring_events(sims):
    for g in sims:
        D = g.sim.drives
        for side, score in ((0, g.sim.home_score), (1, g.sim.away_score)):
            def count(outcome, own):
                m = (D["outcome"] == OUT_IDX[outcome]) & ((D["side"] == side) if own else (D["side"] != side))
                return np.bincount(D["sim"][m], minlength=g.n_sims)
            td = count("TD", True) + count("OPP_TD", False)
            base = 6 * td + 3 * count("FG", True) + 2 * count("SAFETY", False)
            pat = score - base
            assert (pat >= 0).all() and (pat <= 2 * td).all()


def test_out_players_excluded(ctx):
    """Players with final status Out/Doubtful never appear on the pregame roster."""
    from nflsim.players import _status_out
    out = _status_out([2024])
    ro = ctx.roles.merge(ctx.games[["game_id", "season", "week"]], on="game_id")
    ro = ro[ro.season == 2024]
    hit = [(s, w, t, p) in out for s, w, t, p in zip(ro.season, ro.week, ro.team, ro.player_id)]
    assert not any(hit)


def test_correlation_emerges(sims):
    """QB passing yards and his top target's receiving yards move together."""
    g = sims[0]
    for side in (0, 1):
        tp = g.players.teams[side]
        st = g.players.stats[side]
        top = int(np.argmax(tp.tgt_share))
        r = np.corrcoef(st["pass_yards"][:, tp.qb_idx], st["rec_yards"][:, top])[0, 1]
        assert r > 0.25


def test_qb_exits_happen_and_hurt_passing(sims):
    """Starter exits occur at roughly the 2018-20 rate and the backup takes over."""
    rates = []
    for g in sims:
        D = g.sim.drives
        for side in (0, 1):
            ex = np.bincount(D["sim"][(D["side"] == side) & D["qb_out"]], minlength=g.n_sims) > 0
            rates.append(ex.mean())
            b = g.players.teams[side].qb2_idx
            if b >= 0 and ex.any():
                assert g.players.stats[side]["pass_att"][ex, b].sum() > 0
    assert 0.04 < np.mean(rates) < 0.25
