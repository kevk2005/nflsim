"""Streamlit app smoke test on the saved data (no API calls): it loads, shows a
game's markets (all books pooled as one), prices a QB + receiver parlay, and the
greyed-out offerings cannot be selected.
(The QB + own WR/TE warning logic is unit-tested in test_app_support.py.)"""
from pathlib import Path

import pytest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.fixture(scope="module")
def at():
    from nflsim.ingest import PROCESSED
    if not (PROCESSED / "prop_snapshots.parquet").exists() or \
            not (PROCESSED / "drive_plays.parquet").exists():
        pytest.skip("needs saved odds snapshots and player tables")
    from streamlit.testing.v1 import AppTest
    t = AppTest.from_file(APP, default_timeout=600)
    t.run()
    return t


def test_app_loads_and_prices_qb_wr_parlay(at):
    assert not at.exception, at.exception
    # 2026 week 1 NE @ SEA: backfilled snapshot with all 9 prop markets (played game)
    sb = at.sidebar.selectbox[0]
    idx = next((i for i, o in enumerate(sb.options) if "2026 wk1: NE @ SEA" in str(o)), None)
    if idx is None:
        pytest.skip("2026 week 1 NE @ SEA snapshot not saved")
    sb.set_value(idx)                     # option values are row positions in the game list
    at.run()
    assert not at.exception, at.exception
    assert not any(sb.label == "Book" for sb in at.sidebar.selectbox)   # no book picker: books pooled
    # a priceable QB passing-yards over plus a receiving-yards over
    boxes = [c for c in at.checkbox if not c.disabled]
    assert boxes, "no priceable legs on the board"
    qb = next((c for c in boxes if "passing yards over" in c.label), None)
    if qb is None:
        pytest.skip("default game has no QB passing-yards prop")
    qb.check()
    rec = next((c for c in boxes if "receiving yards over" in c.label), None)
    if rec is not None:
        rec.check()
    at.sidebar.slider[0].set_value(0)                            # skip bootstrap for speed
    at.sidebar.select_slider[0].set_value(2000)
    at.run()
    at.button[0].click()                                         # Run simulation
    at.run()
    assert not at.exception, at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert "Joint P (K/N)" in metrics and "Independence benchmark" in metrics
    assert 0 <= float(metrics["Joint P (K/N)"]) <= 1


def test_greyed_out_cannot_be_priced(at):
    dis = [c for c in at.checkbox if c.disabled]
    for c in dis:
        assert c.value is False


def test_tail_scanner_tab(at):
    from nflsim import gamereport as GR
    from nflsim import tails as TL
    assert not at.exception, at.exception
    assert any("Tail scanner" in str(t.label) for t in at.tabs)
    banners = [e.value for e in list(at.error) + list(at.success) + list(at.warning)]
    if TL.load_report() is not None:
        assert any("Backtest keep test" in b for b in banners)
    else:
        assert any("No game-report backtest" in b for b in banners)
    if GR.saved_reports():                    # game report rendered: team section + players table
        md = " ".join(m.value for m in at.markdown)
        assert "#### Game" in md and "#### Players" in md
