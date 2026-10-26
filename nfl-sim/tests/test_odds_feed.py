"""Odds feed tests. No test calls the real API: every network call goes through a
fake transport serving the TEST-ONLY fixtures in tests/fixtures/odds/, and the
real transport is replaced by one that fails the test if it is ever reached."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nflsim import odds as O
from nflsim import odds_feed as F
from nflsim.anchor import get_line
from nflsim.markets import Leg

FIX = Path(__file__).parent / "fixtures" / "odds"
FAKE_KEY = "TEST-KEY-NOT-REAL-123"


def fixture(name):
    return json.loads((FIX / name).read_text())


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Redirect every file the feed writes into tmp_path; forbid the real network."""
    def boom(*a, **k):
        raise AssertionError("a test tried to call the real Odds API")
    monkeypatch.setattr(F, "_real_transport", boom)
    monkeypatch.setenv("ODDS_API_KEY", FAKE_KEY)
    for name, sub in [("ODDS_DIR", "odds"), ("LEDGER", "odds/ledger.jsonl"),
                      ("BACKFILL_STATE", "odds/backfill_state.json"),
                      ("APPROVALS_USED", "odds/historical_approvals_used.json"),
                      ("LINES_PQ", "odds_snapshots.parquet"), ("PROPS_PQ", "prop_snapshots.parquet"),
                      ("CONSENSUS_PQ", "odds_consensus.parquet"),
                      ("UNMATCHED_GAMES", "unmatched_games.csv"),
                      ("UNMATCHED_PLAYERS", "unmatched_players.csv")]:
        monkeypatch.setattr(F, name, tmp_path / sub)
    return tmp_path


@pytest.fixture(scope="module")
def sched():
    from nflsim.ingest import PROCESSED
    if not (PROCESSED / "games.parquet").exists():
        pytest.skip("processed data missing")
    return F.load_schedule()


@pytest.fixture(scope="module")
def mapper():
    return F.PlayerMapper()


class FakeTransport:
    """Serves fixtures by endpoint and returns usage headers like the real API."""

    def __init__(self, remaining=5000, cost=None, empty_first=False):
        self.calls, self.remaining, self.cost = [], remaining, cost or {}
        self.empty_first = empty_first

    def __call__(self, url):
        self.calls.append(url)
        path = url.split("?")[0].replace(F.BASE, "")
        if path == "/v4/sports":
            body, last = [], 0
        elif path.endswith("/events") and "historical" not in path:
            body, last = [{"id": "testevent_bal_kc", "home_team": "Kansas City Chiefs",
                           "away_team": "Baltimore Ravens", "commence_time": "2024-09-06T00:20:00Z"}], 0
        elif "/historical/" in path and "/events/" in path:
            if self.empty_first and not any("historical_props_served" in c for c in self.calls[:-1]):
                self.calls[-1] += "#historical_props_served"
                body, last = fixture("TESTONLY_historical_event_props_empty.json"), 0
            else:
                body, last = fixture("TESTONLY_historical_event_props.json"), 30
        elif "/historical/" in path:
            body, last = fixture("TESTONLY_historical_odds.json"), 30
        elif "/events/" in path:
            body, last = fixture("TESTONLY_event_props.json"), 3
        else:
            body, last = fixture("TESTONLY_live_odds.json"), 3
        last = self.cost.get(path, last)
        self.remaining -= last
        return 200, json.dumps(body).encode(), {"x-requests-last": str(last),
                                               "x-requests-remaining": str(self.remaining),
                                               "x-requests-used": str(20000 - self.remaining)}


def client_with(tmp, transport, approval=None, **caps):
    b = F.Budget(ledger_path=tmp / "odds" / "ledger.jsonl", historical_approval=approval,
                 approvals_used_path=tmp / "odds" / "historical_approvals_used.json", **caps)
    return F.Client(budget=b, transport=transport, min_interval=0)


def seed_remaining(client, n):
    client.budget.log({"time": "x", "month": F.utcnow().strftime("%Y-%m"), "category": "free",
                       "endpoint": "/v4/sports", "estimated": 0, "actual": 0, "remaining": n})


# --------------------------------------------------------------------------- parsing + mapping
def test_parse_lines_and_game_mapping(sched):
    df, um = F.normalize_lines(fixture("TESTONLY_live_odds.json"), "p1", pd.Timestamp("2024-09-05T23:30Z"),
                               sched)
    assert set(df[df.event_id == "testevent_bal_kc"].game_id) == {"2024_01_BAL_KC"}
    assert len(df[df.event_id == "testevent_bal_kc"]) == 3 * 6        # 3 books x (2+2+2) outcomes
    assert df[df.event_id == "testevent_unmatched"].game_id.isna().all()
    assert len(um) == 1 and "no nflverse game" in um[0]["reason"]   # logged, never guessed
    sp = df[(df.market == "spreads") & (df.book == "bookA") & (df.team == "KC")]
    assert sp.point.iloc[0] == -3.0 and sp.price.iloc[0] == -110


def test_map_game_rules(sched):
    gid, why = F.map_game("Kansas City Chiefs", "Baltimore Ravens", "2024-09-06T00:20:00Z", sched)
    assert gid == "2024_01_BAL_KC" and why == "ok"            # 8:20 pm ET Sept 5 -> ET date Sept 5
    gid, why = F.map_game("Baltimore Ravens", "Kansas City Chiefs", "2024-09-06T00:20:00Z", sched)
    assert gid == "2024_01_BAL_KC" and why == "ok_swapped_home_away"
    assert F.map_game("Kansas City Chiefs", "Baltimore Ravens", "2024-09-08T17:00:00Z", sched)[0] is None
    assert F.map_game("Springfield Atoms", "Baltimore Ravens", "2024-09-06T00:20:00Z", sched)[0] is None


def test_player_mapping(mapper):
    t = ("KC", "BAL")
    assert mapper.match("Patrick Mahomes", "2024_01_BAL_KC", 2024, t, "passing_yards")[0] == "00-0033873"
    assert mapper.match("Rashee Rice", "2024_01_BAL_KC", 2024, t, "receiving_yards")[0] is not None
    assert mapper.match("Derrick Henry", "2024_01_BAL_KC", 2024, t, "anytime_td")[0] is not None
    pid, why = mapper.match("Fake Testplayer", "2024_01_BAL_KC", 2024, t, "receiving_yards")
    assert pid is None and "no name match" in why
    # a real player on neither team must not be matched (team constraint)
    assert mapper.match("Josh Allen", "2024_01_BAL_KC", 2024, t, "passing_yards")[0] is None
    assert F.norm_name("Odell Beckham Jr.") == F.norm_name("odell beckham")


def test_props_parse_devig(mapper):
    df, um = F.normalize_props(fixture("TESTONLY_event_props.json"), "pp", pd.Timestamp("2024-09-05T23:30Z"),
                               "2024_01_BAL_KC", 2024, mapper)
    assert {r["raw_name"] for r in um} == {"Fake Testplayer"}
    mah = df[(df.player_raw == "Patrick Mahomes")]
    assert set(mah.side) == {"over", "under"} and mah.player_id.iloc[0] == "00-0033873"
    pa, pb, _ = O.devig(-115, -105)
    assert np.isclose(mah[mah.side == "over"].fair_prob.iloc[0], pa)
    assert np.isclose(mah.fair_prob.sum(), 1)
    one_sided = df[(df.book == "bookB") & (df.market == "anytime_td")]
    assert one_sided.fair_prob.isna().all()                       # no 'No' price -> no de-vig
    matched = df.loc[df.player_id.notna(), "player_raw"].nunique() / df.player_raw.nunique()
    assert np.isclose(matched, 3 / 4)


def test_consensus(sched):
    df, _ = F.normalize_lines(fixture("TESTONLY_live_odds.json"), "p1", pd.Timestamp("2024-09-05T23:30Z"),
                              sched)
    c = F.consensus(df).set_index("game_id").loc["2024_01_BAL_KC"]
    assert c.home_spread == -3.0 and c.total == 46.5               # medians of (-3,-2.5,-3.5), (46.5,47,46)
    ph = np.median([O.devig(-150, 130)[0], O.devig(-140, 120)[0], O.devig(-160, 135)[0]])
    assert np.isclose(c.p_home_ml, ph) and c.n_books_ml == 3


# --------------------------------------------------------------------------- budget
def test_budget_caps_reserve_and_unknown(tmp_path):
    b = F.Budget(cap_lines=10, cap_props=50, reserve=2000, ledger_path=tmp_path / "l.jsonl")
    with pytest.raises(F.BudgetRefused, match="unknown"):
        b.check("lines", 3)                                        # no remaining known yet
    b.check("free", 0)                                             # free calls always allowed
    b.log({"month": F.utcnow().strftime("%Y-%m"), "category": "lines", "estimated": 3, "actual": 9,
           "remaining": 10000})
    with pytest.raises(F.BudgetRefused, match="lines cap"):
        b.check("lines", 3)                                        # 9 used + 3 > 10 (actual counts)
    b.check("props", 50)
    with pytest.raises(F.BudgetRefused, match="props cap"):
        b.check("props", 51)
    b.log({"month": F.utcnow().strftime("%Y-%m"), "category": "free", "estimated": 0, "actual": 0,
           "remaining": 2010})
    b.check("props", 10)                                           # 2010 - 10 = 2000: allowed
    with pytest.raises(F.BudgetRefused, match="reserve"):
        b.check("props", 11)                                       # would dip below the reserve
    # last month's usage does not count against this month
    b.log({"month": "1999-01", "category": "props", "estimated": 0, "actual": 999, "remaining": 2010})
    assert b.used("props") == 0


def test_client_refuses_before_calling_and_logs_without_key(isolated):
    t = FakeTransport()
    c = client_with(isolated, t, cap_lines=2)
    seed_remaining(c, 5000)
    with pytest.raises(F.BudgetRefused):
        F.pull_lines(client=c)                                     # est 3 > cap 2
    assert t.calls == []                                           # refused before any call
    c.budget.cap_lines = 1000
    F.pull_lines(client=c)
    assert len(t.calls) == 1
    led = (isolated / "odds" / "ledger.jsonl").read_text()
    assert FAKE_KEY not in led and "apiKey" not in led             # key never logged
    e = json.loads(led.splitlines()[-1])
    assert e["estimated"] == 3 and e["actual"] == 3 and e["remaining"] == 4997
    for p in (isolated / "odds").rglob("*.json"):
        assert FAKE_KEY not in p.read_text()                       # nor saved in raw files


def test_alternate_prop_keys_parse_to_base_market(mapper):
    ev = fixture("TESTONLY_event_props.json")
    ev = json.loads(json.dumps(ev))
    ev["bookmakers"][0]["markets"].append(
        {"key": "player_pass_yds_alternate", "outcomes": [
            {"name": "Over", "description": "Patrick Mahomes", "price": 210, "point": 299.5}]})
    df, _ = F.normalize_props(ev, "pp", pd.Timestamp("2024-09-05T23:30Z"), "2024_01_BAL_KC", 2024, mapper)
    alt = df[df.market_key == "player_pass_yds_alternate"]
    assert len(alt) == 1 and alt.market.iloc[0] == "passing_yards" and alt.player_id.iloc[0] == "00-0033873"
    assert alt.fair_prob.isna().all()


def test_dry_run_makes_no_calls(isolated, sched):
    t = FakeTransport()
    c = client_with(isolated, t)
    r = F.pull_lines(client=c, dry_run=True)
    assert r["dry_run"] and r["est_total"] == 3
    r = F.pull_props(client=c, dry_run=True, sched=sched)
    assert r["dry_run"] and r["est_total"] == 9 * r["n_games"]
    r = F.pull_props(client=c, dry_run=True, sched=sched, alternates=True)
    assert r["dry_run"] and r["alternates"] and r["est_total"] == 17 * r["n_games"]
    r = F.pull_lines(client=c, dry_run=True, bookmaker="fanatics")
    assert r["calls"][0]["bookmakers"] == "fanatics" and r["est_total"] == 3
    assert t.calls == [] and not (isolated / "odds" / "ledger.jsonl").exists()


def test_live_pulls_store_snapshots(isolated, sched, mapper, monkeypatch):
    t = FakeTransport()
    c = client_with(isolated, t)
    seed_remaining(c, 5000)
    r = F.pull_lines(client=c, sched=sched)
    assert r["events"] == 2 and len(r["unmatched_games"]) == 1
    assert F.LINES_PQ.exists() and F.CONSENSUS_PQ.exists()
    raws = list((isolated / "odds" / "lines").glob("*.json"))
    assert len(raws) == 1 and "pulled_at" in json.loads(raws[0].read_text())
    # props: pretend 'now' is just before the 2024 opener so it is "upcoming"
    monkeypatch.setattr(F, "utcnow", lambda: pd.Timestamp("2024-09-05T20:00Z"))
    r = F.pull_props(client=c, sched=sched, mapper=mapper, days=2)
    assert [g["game_id"] for g in r["games"]] == ["2024_01_BAL_KC"]
    assert r["games"][0]["matched_players"] == 3 and r["games"][0]["players"] == 4
    assert {u["raw_name"] for u in r["unmatched_players"]} == {"Fake Testplayer"}


# --------------------------------------------------------------------------- integration rules
def _write_consensus(rows):
    pd.DataFrame(rows).to_parquet(F.CONSENSUS_PQ, index=False)


def _crow(pulled, spread=-3.0, total=46.5, source="live"):
    return {"pull_id": f"p{pulled}", "game_id": "G", "source": source, "pulled_at": pd.Timestamp(pulled),
            "bookmaker_filter": None, "commence_time": pd.Timestamp("2024-09-06T00:20Z"),
            "home_team": "KC", "home_spread": spread, "total": total, "p_home_ml": 0.6,
            "n_books_spread": 3, "n_books_total": 3, "n_books_ml": 3}


def test_no_post_kickoff_snapshot():
    ko = pd.Timestamp("2024-09-06T00:20Z")
    _write_consensus([_crow("2024-09-05T22:00Z", -3.0), _crow("2024-09-06T01:00Z", -10.0)])   # 2nd is after kickoff
    s = F.latest_consensus("G", ko, as_of=pd.Timestamp("2030-01-01T00:00Z"))
    assert s["home_spread"] == -3.0
    _write_consensus([_crow("2024-09-06T00:20Z", -10.0)])          # exactly at kickoff: not allowed
    assert F.latest_consensus("G", ko, as_of=pd.Timestamp("2030-01-01T00:00Z")) is None


def test_stale_warning_and_line_priority():
    ko = pd.Timestamp("2024-09-06T00:20Z")
    _write_consensus([_crow("2024-09-05T10:00Z", -3.0, 46.5)])
    s = F.latest_consensus("G", ko, as_of=ko)
    assert s["stale"] and s["age_hours"] > 6
    row = pd.Series({"season": 2024, "mkt_spread_line": 2.0, "mkt_total_line": 48.0, "label_home_score": 1})
    ln = get_line(row, snapshot=s)
    assert ln.target_margin == 3.0 and ln.target_total == 46.5 and "Odds API" in ln.source
    assert "WARNING" in ln.warning
    ln = get_line(row, home_spread=-7.0, snapshot=s)                # user beats snapshot
    assert ln.target_margin == 7.0 and ln.target_total == 46.5
    ln = get_line(row, snapshot=None)                               # nflverse when no snapshot
    assert ln.target_margin == 2.0 and "nflverse" in ln.source and ln.warning == ""
    _write_consensus([_crow("2024-09-05T22:00Z")])
    assert not F.latest_consensus("G", ko, as_of=ko)["stale"]


def test_lookup_price_best_book_and_kickoff(isolated, mapper):
    df, _ = F.normalize_props(fixture("TESTONLY_event_props.json"), "pp", pd.Timestamp("2024-09-05T23:30Z"),
                              "2024_01_BAL_KC", 2024, mapper)
    late, _ = F.normalize_props(fixture("TESTONLY_event_props.json"), "late", pd.Timestamp("2024-09-06T01:00Z"),
                                "2024_01_BAL_KC", 2024, mapper)
    late["price"] = 500                                            # post-kickoff prices must be ignored
    F.store_props(pd.concat([df, late]))
    ko = pd.Timestamp("2024-09-06T00:20Z")
    rice = df[df.player_raw == "Rashee Rice"].player_id.iloc[0]
    leg = Leg("receiving_yards", "over", 55.5, rice)
    p = F.lookup_price(leg, "2024_01_BAL_KC", ko, "KC", as_of=pd.Timestamp("2030-01-01T00:00Z"))
    assert p["book"] == "bookB" and p["price"] == -105 and p["n_books_quoting"] == 2
    assert F.lookup_price(leg, "2024_01_BAL_KC", ko, "KC", book="bookA")["price"] == -110
    assert F.lookup_price(Leg("receiving_yards", "over", 60.5, rice), "2024_01_BAL_KC", ko, "KC") is None


# --------------------------------------------------------------------------- live-only rules
def test_historical_endpoints_refused_without_approval(isolated):
    t = FakeTransport()
    c = client_with(isolated, t)                                   # no approval
    seed_remaining(c, 5000)
    path = f"/v4/historical/sports/{F.SPORT}/odds"
    with pytest.raises(F.BudgetRefused, match="disabled"):
        c.get(path, {"date": "2025-01-01T00:00:00Z"}, "props", 30)  # not via the backfill
    with pytest.raises(F.BudgetRefused, match="cap is 0"):
        c.get(path, {"date": "2025-01-01T00:00:00Z"}, "historical", 30, allow_historical=True)
    month = F.utcnow().strftime("%Y-%m")
    c2 = client_with(isolated, t, approval="1999-01:14000")        # approval for another month
    seed_remaining(c2, 5000)
    assert c2.budget.cap("historical") == 0
    c3 = client_with(isolated, t, approval=f"{month}:14000")
    assert c3.budget.cap("historical") == 14000
    c3.budget.consume_approval()
    assert c3.budget.cap("historical") == 0                         # one-time
    assert t.calls == []
    b = F.Budget()
    assert (b.cap_lines, b.cap_props, b.reserve) == (2000, 16000, 2000)


def test_bookmaker_filter(isolated, sched, mapper, monkeypatch):
    t = FakeTransport()
    c = client_with(isolated, t)
    seed_remaining(c, 5000)
    r = F.pull_lines(client=c, sched=sched, bookmaker="fanatics")
    q = t.calls[-1].split("?")[1]
    assert "bookmakers=fanatics" in q and "regions=" not in q        # bookmakers replaces regions
    led = json.loads((isolated / "odds" / "ledger.jsonl").read_text().splitlines()[-1])
    assert led["estimated"] == 3                                       # same cost as the us region
    assert r["pull_id"].endswith("_fanatics")
    lines = pd.read_parquet(F.LINES_PQ)
    assert (lines.bookmaker_filter == "fanatics").all()
    monkeypatch.setattr(F, "utcnow", lambda: pd.Timestamp("2024-09-05T20:00Z"))
    F.pull_props(client=c, sched=sched, mapper=mapper, days=2, bookmaker="fanatics")
    assert "bookmakers=fanatics" in t.calls[-1]
    with pytest.raises(ValueError, match="unknown bookmaker"):
        F.pull_lines(client=c, dry_run=True, bookmaker="notabook")


def test_every_live_pull_is_kept(isolated, sched, monkeypatch):
    """Each pull is saved under its own timestamp, building our own line history."""
    t = FakeTransport()
    c = client_with(isolated, t)
    seed_remaining(c, 5000)
    clock = {"now": pd.Timestamp("2024-09-04T12:00Z")}
    monkeypatch.setattr(F, "utcnow", lambda: clock["now"])
    F.pull_lines(client=c, sched=sched)
    clock["now"] = pd.Timestamp("2024-09-05T12:00Z")
    F.pull_lines(client=c, sched=sched)
    lines = pd.read_parquet(F.LINES_PQ)
    assert lines.pulled_at.nunique() == 2 and lines.pull_id.nunique() == 2
    assert len(list((isolated / "odds" / "lines").glob("*.json"))) == 2
    cons = pd.read_parquet(F.CONSENSUS_PQ)
    assert len(cons[cons.game_id == "2024_01_BAL_KC"]) == 2           # one consensus row per pull


# --------------------------------------------------------------------------- one-time backfill
TODAY = pd.Timestamp("2024-09-09T12:00Z")       # tests reuse the 2024 opener as a fake "season"


@pytest.fixture
def season_2024(monkeypatch):
    monkeypatch.setattr(F, "BACKFILL_SEASONS", (2024,))


def _approved_client(isolated, t, cap=14000):
    c = client_with(isolated, t, approval=f"{F.utcnow().strftime('%Y-%m')}:{cap}")
    seed_remaining(c, 100000)
    return c


def test_backfill_dry_run_and_confirm(isolated, sched, season_2024):
    t = FakeTransport(remaining=100000)
    c = _approved_client(isolated, t)
    r = F.backfill(client=c, dry_run=True, sched=sched, today=TODAY)
    # 2024 week 1 before Mon 9 Sept: 15 games (the Monday-night game is "today", excluded)
    assert r["dry_run"] and r["games_in_scope"] == 15 and t.calls == []
    assert r["est_total"] == 30 * r["kickoff_slots"] + 90 * 15
    with pytest.raises(F.BudgetRefused, match="--confirm"):
        F.backfill(client=c, sched=sched, today=TODAY)              # no spending without --confirm
    assert t.calls == []


def test_backfill_ten_books_one_request_and_snapshot_times(isolated, sched, mapper, season_2024):
    t = FakeTransport(remaining=100000)
    c = _approved_client(isolated, t)
    r = F.backfill(client=c, confirm=True, sched=sched, mapper=mapper, today=TODAY, max_calls=2)
    assert r["done"] == 2 and len(t.calls) == 2
    for u in t.calls:
        q = u.split("?")[1]
        assert "bookmakers=" + "%2C".join(F.BACKFILL_BOOKS) in q and "regions=" not in q
        assert "fanatics" in q
        assert "date=2024-09-06T00%3A19%3A00Z" in q                # kickoff 00:20Z - 1 min
    led = [json.loads(l) for l in (isolated / "odds" / "ledger.jsonl").read_text().splitlines()]
    assert [e["estimated"] for e in led if e["category"] == "historical"] == [30, 90]
    lines, props = pd.read_parquet(F.LINES_PQ), pd.read_parquet(F.PROPS_PQ)
    assert (lines.snapshot_ts == pd.Timestamp("2024-09-06T00:15:00Z")).all()   # API's own time, every row
    assert (props.snapshot_ts == pd.Timestamp("2024-09-06T00:15:00Z")).all()
    assert (F.snapshot_time(props) < pd.Timestamp("2024-09-06T00:20Z")).all()  # before kickoff


def test_backfill_resume_idempotent_cap_and_one_time(isolated, sched, mapper, season_2024):
    state_p = isolated / "odds" / "backfill_state.json"
    t = FakeTransport(remaining=100000)
    c = _approved_client(isolated, t, cap=150)
    F.backfill(client=c, confirm=True, sched=sched, mapper=mapper, today=TODAY, max_calls=2)
    n = len(t.calls)
    # rerun: already-saved slot and game are skipped (never paid twice)
    plan = F.backfill_plan(sched, F._load_state(state_p), TODAY)
    assert all(p.get("game_id") != "2024_01_BAL_KC" and p["slot"] != "2024-09-06T00:20:00Z" for p in plan)
    # lost state but raw files kept -> re-parsed from disk, no new paid call
    state_p.unlink()
    F.backfill(client=c, confirm=True, sched=sched, mapper=mapper, today=TODAY, max_calls=2)
    assert len(t.calls) == n
    # cap stop: 60 billed of 150; next slot lines (30) fits, next props est 90 does not
    r = F.backfill(client=c, confirm=True, sched=sched, mapper=mapper, today=TODAY)
    assert r["stopped"] and "historical cap" in r["stopped"] and r["calls_remaining"] > 0
    assert c.budget.used("historical") <= 150
    assert c.budget.cap("historical") == 150                         # not consumed: unfinished


def test_backfill_completion_consumes_approval(isolated, sched, mapper, season_2024, monkeypatch):
    t = FakeTransport(remaining=100000)
    c = _approved_client(isolated, t)
    # limit scope to the opener only, so the run completes
    real = F.backfill_games
    monkeypatch.setattr(F, "backfill_games", lambda s_, today=None: real(s_, today).head(1))
    r = F.backfill(client=c, confirm=True, sched=sched, mapper=mapper, today=TODAY)
    assert r["calls_remaining"] == 0 and "consumed" in r["approval"]
    assert c.budget.cap("historical") == 0
    with pytest.raises(F.BudgetRefused, match="cap is 0"):
        c.get(f"/v4/historical/sports/{F.SPORT}/odds", {"date": "x"}, "historical", 30, allow_historical=True)


def test_backfill_empty_snapshot_steps_back_free(isolated, sched, mapper, season_2024, monkeypatch):
    t = FakeTransport(remaining=100000, empty_first=True)
    c = _approved_client(isolated, t)
    real = F.backfill_games
    monkeypatch.setattr(F, "backfill_games", lambda s_, today=None: real(s_, today).head(1))
    F.backfill(client=c, confirm=True, sched=sched, mapper=mapper, today=TODAY)
    props_calls = [u for u in t.calls if "/events/" in u]
    assert len(props_calls) == 2 and "date=2024-09-06T00%3A10%3A00Z" in props_calls[1]  # previous_timestamp
    assert c.budget.used("historical") == 30 + 0 + 30                # the empty one billed 0


def test_cost_overrun_stops_run(isolated, sched, mapper, season_2024):
    t = FakeTransport(remaining=100000, cost={f"/v4/historical/sports/{F.SPORT}/odds": 300})
    c = _approved_client(isolated, t)
    r = F.backfill(client=c, confirm=True, sched=sched, mapper=mapper, today=TODAY)
    assert "billed 300 > estimated 30" in r["stopped"] and len(t.calls) == 1
    assert (isolated / "odds" / "historical_lines").exists()        # what was paid for is kept
