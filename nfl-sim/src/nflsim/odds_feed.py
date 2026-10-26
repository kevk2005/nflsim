"""The Odds API feed: live game lines and player props for UPCOMING games only,
with a hard credit budget. Every live pull is saved with its UTC pull time, so the
project builds its own line history going forward. No historical endpoint is ever
called (Client refuses any /historical/ path).

Facts below are from the official v4 docs (https://the-odds-api.com/liveapi/guides/v4/,
/sports-odds-data/betting-markets.html, /sports-odds-data/bookmaker-apis.html), read 2026-10-26:
  * NFL sport key: americanfootball_nfl
  * GET /v4/sports/{sport}/odds              cost = markets x regions (0 if no events)
  * GET /v4/sports/{sport}/events            free
  * GET /v4/sports/{sport}/events/{id}/odds  cost = unique markets RETURNED x regions
                                             (player props only via this endpoint)
  * GET /v4/sports                           free
  * `bookmakers` param: "If both bookmakers and regions are specified, bookmakers takes
    priority"; "Every group of 10 bookmakers is the equivalent of 1 region". So one
    bookmaker costs the same as the whole `us` region.
  * Usage headers: x-requests-remaining, x-requests-used, x-requests-last
Estimates here are upper bounds (all requested markets returned); the actual cost
is read from x-requests-last after every call.

The API key is read from ODDS_API_KEY (environment or nfl-sim/.env). It is never
printed, logged or written to disk: logged URLs have the key removed.

CLI:  python -m nflsim.odds_feed {pull-lines,pull-props,latest,credits} [--dry-run] [--bookmaker KEY]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import odds as O
from .ingest import PROCESSED, RAW, ROOT

BASE = "https://api.the-odds-api.com"
SPORT = "americanfootball_nfl"
REGIONS = "us"
GAME_MARKETS = ["h2h", "spreads", "totals"]
# markets.py name -> The Odds API market key (from the betting-markets docs)
PROP_MARKETS = {
    "passing_yards": "player_pass_yds",
    "passing_tds": "player_pass_tds",
    "pass_attempts": "player_pass_attempts",
    "completions": "player_pass_completions",
    "rushing_yards": "player_rush_yds",
    "rush_attempts": "player_rush_attempts",
    "receptions": "player_receptions",
    "receiving_yards": "player_reception_yds",
    "anytime_td": "player_anytime_td",
}
KEY_TO_MARKET = {v: k for k, v in PROP_MARKETS.items()}
# Bookmaker keys in the docs' US regions (bookmaker-apis page). --bookmaker must be one of these.
US_BOOKMAKERS = {
    "us": ["betonlineag", "betmgm", "betrivers", "betus", "bovada", "williamhill_us", "draftkings",
           "fanatics", "fanduel", "lowvig", "mybookieag"],
    "us2": ["ballybet", "betanysports", "betparx", "courtside", "espnbet", "fliff", "hardrockbet",
            "hardrockbet_az", "hardrockbet_fl", "hardrockbet_oh", "rebet"],
}
KNOWN_BOOKMAKERS = {b for v in US_BOOKMAKERS.values() for b in v}
STALE_HOURS = 6.0
ET = ZoneInfo("America/New_York")

ODDS_DIR = RAW / "odds"
LEDGER = ODDS_DIR / "ledger.jsonl"
BACKFILL_STATE = ODDS_DIR / "backfill_state.json"
APPROVALS_USED = ODDS_DIR / "historical_approvals_used.json"
LINES_PQ = PROCESSED / "odds_snapshots.parquet"
PROPS_PQ = PROCESSED / "prop_snapshots.parquet"
CONSENSUS_PQ = PROCESSED / "odds_consensus.parquet"
UNMATCHED_GAMES = PROCESSED / "odds_unmatched_games.csv"
UNMATCHED_PLAYERS = PROCESSED / "odds_unmatched_players.csv"

TEAM_NAMES = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
    # former names (relocations / renames)
    "Washington Football Team": "WAS", "Washington Redskins": "WAS", "Oakland Raiders": "LV",
    "San Diego Chargers": "LAC", "St. Louis Rams": "LA", "St Louis Rams": "LA",
}
ABBR_TO_NAME = {v: k for k, v in TEAM_NAMES.items() if k not in
                ("Washington Football Team", "Washington Redskins", "Oakland Raiders",
                 "San Diego Chargers", "St. Louis Rams", "St Louis Rams")}


def utcnow() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _iso(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y%m%dT%H%M%SZ")


# --------------------------------------------------------------------------- #
# Key
# --------------------------------------------------------------------------- #
def load_api_key() -> str | None:
    k = os.environ.get("ODDS_API_KEY")
    if k:
        return k.strip()
    envf = ROOT / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            line = line.strip()
            if line.startswith("ODDS_API_KEY=") and len(line) > len("ODDS_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            if line.strip().startswith(name + "="):
                v = line.split("=", 1)[1].strip() or None
    return int(v) if v else default


# --------------------------------------------------------------------------- #
# Budget and ledger
# --------------------------------------------------------------------------- #
CATEGORIES = ("lines", "props", "historical", "free")


class BudgetRefused(Exception):
    pass


class ApiError(Exception):
    pass


class CostOverrun(Exception):
    """A call was billed more than its estimate. Raised after the response is
    returned and logged; callers save what they paid for and stop."""

    def __init__(self, msg, data=None, headers=None):
        super().__init__(msg)
        self.data, self.headers = data, headers


@dataclass
class Budget:
    """Monthly caps by category (calendar month, UTC). The reserve is never spent
    automatically: a call is refused if it would leave fewer than `reserve`
    credits remaining (per the API's own x-requests-remaining header)."""
    cap_lines: int = 2000
    cap_props: int = 16000
    reserve: int = 2000
    ledger_path: Path = LEDGER
    # One-time historical allowance "YYYY-MM:CAP" (ODDS_HISTORICAL_APPROVAL in .env).
    # The historical cap is CAP only during that UTC month and only until the approved
    # backfill completes; otherwise 0. A new run needs a new, explicit approval.
    historical_approval: str | None = None
    approvals_used_path: Path = APPROVALS_USED

    @classmethod
    def from_env(cls, ledger_path: Path = LEDGER) -> "Budget":
        appr = os.environ.get("ODDS_HISTORICAL_APPROVAL")
        if appr is None and (ROOT / ".env").exists():
            for line in (ROOT / ".env").read_text().splitlines():
                if line.strip().startswith("ODDS_HISTORICAL_APPROVAL="):
                    appr = line.split("=", 1)[1].strip() or None
        return cls(_env_int("ODDS_CAP_LINES", 2000), _env_int("ODDS_CAP_PROPS", 16000),
                   _env_int("ODDS_RESERVE", 2000), ledger_path, appr, APPROVALS_USED)

    def approvals_used(self) -> list:
        p = self.approvals_used_path
        return json.loads(p.read_text()) if p.exists() else []

    def historical_cap(self) -> int:
        a = self.historical_approval
        if not a or ":" not in a or a in self.approvals_used():
            return 0
        month, cap = a.split(":", 1)
        return int(cap) if month == utcnow().strftime("%Y-%m") else 0

    def consume_approval(self) -> None:
        if self.historical_approval and self.historical_approval not in self.approvals_used():
            self.approvals_used_path.parent.mkdir(parents=True, exist_ok=True)
            self.approvals_used_path.write_text(json.dumps(self.approvals_used() + [self.historical_approval]))

    def cap(self, category: str) -> int:
        if category == "historical":
            return self.historical_cap()
        return {"lines": self.cap_lines, "props": self.cap_props, "free": 0}[category]

    def entries(self) -> list[dict]:
        if not self.ledger_path.exists():
            return []
        return [json.loads(l) for l in self.ledger_path.read_text().splitlines() if l.strip()]

    def used(self, category: str, month: str | None = None) -> int:
        month = month or utcnow().strftime("%Y-%m")
        tot = 0
        for e in self.entries():
            if e.get("month") == month and e.get("category") == category and not e.get("dry_run"):
                a = e.get("actual")
                tot += int(a) if a is not None else int(e.get("estimated") or 0)
        return tot

    def remaining(self) -> int | None:
        for e in reversed(self.entries()):
            if e.get("remaining") is not None:
                return int(e["remaining"])
        return None

    def check(self, category: str, est: int, pending: int = 0) -> None:
        """Raise BudgetRefused if `est` more credits (after `pending` already planned
        in this command) would break the category cap or the reserve floor."""
        if est <= 0:
            return
        used = self.used(category) + pending
        cap = self.cap(category)
        if used + est > cap:
            raise BudgetRefused(f"{category} cap: used {used} + estimated {est} > cap {cap} this month")
        rem = self.remaining()
        if rem is None:
            raise BudgetRefused("remaining credits unknown; run `python -m nflsim.odds_feed credits "
                                "--refresh` (free call) first")
        if rem - pending - est < self.reserve:
            raise BudgetRefused(f"reserve floor: remaining {rem - pending} - estimated {est} "
                                f"< reserve {self.reserve}")

    def log(self, entry: dict) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")


# --------------------------------------------------------------------------- #
# HTTP client
# --------------------------------------------------------------------------- #
def _ssl_context():
    """Verified TLS using certifi's CA bundle (python.org macOS builds ship without
    system root certificates). Verification is never disabled."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _real_transport(url: str, timeout: float = 30.0):
    req = urllib.request.Request(url, headers={"User-Agent": "nflsim/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
            return r.status, r.read(), {k.lower(): v for k, v in r.headers.items()}
    except urllib.error.HTTPError as e:
        return e.code, e.read(), {k.lower(): v for k, v in (e.headers or {}).items()}
    except (urllib.error.URLError, OSError) as e:
        # never let a traceback carry the URL (and its apiKey) to the console
        raise ApiError(f"network error calling {_redact(url).replace(BASE, '').split('?')[0]}: "
                       f"{getattr(e, 'reason', e)}") from None


def _redact(url: str) -> str:
    return re.sub(r"apiKey=[^&]*", "apiKey=REDACTED", url)


@dataclass
class Client:
    budget: Budget = field(default_factory=Budget.from_env)
    transport: object = None
    key: str | None = None
    min_interval: float = 1.0          # docs: space out requests to avoid 429s
    _last: float = 0.0

    def get(self, path: str, params: dict, category: str, est: int, note: str = "",
            allow_historical: bool = False) -> tuple:
        """One API call with budget check before and ledger entry after. Returns (json, headers).
        Historical endpoints are refused unless called from the approved backfill
        (allow_historical=True) with category 'historical' and a non-zero approved cap."""
        if "/historical/" in path and not (allow_historical and category == "historical"):
            raise BudgetRefused("historical endpoints are disabled (only the approved one-time backfill may call them)")
        if category == "historical" and self.budget.cap("historical") <= 0:
            raise BudgetRefused("historical cap is 0: no active approval (ODDS_HISTORICAL_APPROVAL) for this month, "
                                "or the approved backfill already completed")
        if category not in CATEGORIES:
            raise BudgetRefused(f"unknown budget category {category!r}")
        self.budget.check(category, est)
        key = self.key or load_api_key()
        if not key:
            raise ApiError("ODDS_API_KEY is not set (add it to nfl-sim/.env)")
        transport = self.transport or _real_transport
        q = dict(params, apiKey=key)
        url = f"{BASE}{path}?{urllib.parse.urlencode(q)}"
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0 and self.transport is None:
            time.sleep(wait)
        status, body, headers = transport(url)
        self._last = time.time()

        def h(name):
            v = headers.get(name)
            return int(float(v)) if v not in (None, "") else None

        entry = {"time": _iso(utcnow()), "month": utcnow().strftime("%Y-%m"), "category": category,
                 "endpoint": path, "params": {k: v for k, v in params.items()}, "estimated": est,
                 "actual": h("x-requests-last"), "remaining": h("x-requests-remaining"),
                 "used_total": h("x-requests-used"), "status": status, "note": note}
        self.budget.log(entry)
        if status != 200:
            msg = body.decode("utf-8", "replace")[:300] if isinstance(body, (bytes, bytearray)) else str(body)
            raise ApiError(f"HTTP {status} from {_redact(url).replace(BASE, '')}: {msg}")
        data = json.loads(body)
        if entry["actual"] is not None and entry["actual"] > est:
            raise CostOverrun(f"billed {entry['actual']} > estimated {est} for {path}; stopping", data, headers)
        return data, headers


# --------------------------------------------------------------------------- #
# Schedule / mapping
# --------------------------------------------------------------------------- #
def load_schedule() -> pd.DataFrame:
    g = pd.read_parquet(PROCESSED / "games.parquet")
    t = g["gametime"].fillna("13:00")
    local = pd.to_datetime(g["gameday"].dt.strftime("%Y-%m-%d") + " " + t)
    g["kickoff_utc"] = local.dt.tz_localize(ET, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
    return g


def map_game(home_name: str, away_name: str, commence: str, sched: pd.DataFrame) -> tuple:
    """(game_id or None, reason). Exact match on both teams and the US-Eastern
    calendar date of kickoff. A team pair plays at most once on a date, so a
    home/away swap (neutral-site designation) is matched and flagged, not guessed."""
    h, a = TEAM_NAMES.get(home_name), TEAM_NAMES.get(away_name)
    if h is None or a is None:
        return None, f"unknown team name: {home_name if h is None else away_name}"
    day = pd.Timestamp(commence).tz_convert(ET).normalize().tz_localize(None)
    m = sched[(sched["home_team"] == h) & (sched["away_team"] == a) & (sched["gameday"] == day)]
    if len(m) == 1:
        return m["game_id"].iloc[0], "ok"
    m = sched[(sched["home_team"] == a) & (sched["away_team"] == h) & (sched["gameday"] == day)]
    if len(m) == 1:
        return m["game_id"].iloc[0], "ok_swapped_home_away"
    return None, f"no nflverse game {a}@{h} on {day.date()}"


_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("'", "").replace("’", "").replace("-", " ")
    s = _SUFFIX.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


class PlayerMapper:
    """Map a prop's player name to gsis_id using name + the game's two teams
    (+ position only to break ties). Candidates: that game's pregame roster
    (roles) and the season roster for the two teams. Exact normalized-name
    match only; ambiguous or missing -> None (logged by the caller)."""

    PASS_MARKETS = {"passing_yards", "passing_tds", "pass_attempts", "completions"}

    def __init__(self, roles: pd.DataFrame | None = None, rosters: pd.DataFrame | None = None):
        if roles is None:
            p = PROCESSED / "roles.parquet"
            roles = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["game_id", "team", "player_id", "position"])
        if rosters is None:
            from .ingest import load
            rosters = load("rosters").to_pandas()
        self.roles = roles
        r = rosters[["season", "team", "gsis_id", "full_name", "first_name", "last_name", "football_name",
                     "position"]].dropna(subset=["gsis_id"])
        self.rosters = r
        names = {}
        for row in r.itertuples():
            for nm in {row.full_name, f"{row.first_name} {row.last_name}", f"{row.football_name} {row.last_name}"}:
                if isinstance(nm, str) and "nan" not in nm.lower().split():
                    names.setdefault(norm_name(nm), set()).add(row.gsis_id)
        self.name_index = names
        self.pos = r.drop_duplicates("gsis_id", keep="last").set_index("gsis_id")["position"].to_dict()

    def candidates(self, game_id: str | None, season: int, teams: tuple) -> set:
        ids = set(self.rosters[(self.rosters["season"] == season) & self.rosters["team"].isin(teams)]["gsis_id"])
        if game_id is not None:
            ids |= set(self.roles[self.roles["game_id"] == game_id]["player_id"])
        return ids

    def match(self, raw_name: str, game_id: str | None, season: int, teams: tuple,
              market: str | None = None) -> tuple:
        hits = self.name_index.get(norm_name(raw_name), set()) & self.candidates(game_id, season, teams)
        if len(hits) > 1 and market in self.PASS_MARKETS:
            qb = {h for h in hits if self.pos.get(h) == "QB"}
            hits = qb if len(qb) == 1 else hits
        if len(hits) == 1:
            return next(iter(hits)), "ok"
        return None, ("no name match on either team" if not hits else f"ambiguous: {sorted(hits)}")


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def normalize_lines(events: list, pull_id: str, pulled_at: pd.Timestamp, sched: pd.DataFrame,
                    bookmaker_filter: str | None = None, snapshot_ts=None,
                    source: str = "live") -> tuple[pd.DataFrame, list]:
    rows, unmatched = [], []
    for ev in events:
        gid, why = map_game(ev["home_team"], ev["away_team"], ev["commence_time"], sched)
        if gid is None:
            unmatched.append({"pull_id": pull_id, "event_id": ev.get("id"), "home": ev["home_team"],
                              "away": ev["away_team"], "commence_time": ev["commence_time"], "reason": why})
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk["key"] not in GAME_MARKETS:
                    continue
                for oc in mk.get("outcomes", []):
                    rows.append({
                        "pull_id": pull_id, "source": source, "pulled_at": pulled_at,
                        "snapshot_ts": pd.Timestamp(snapshot_ts) if snapshot_ts is not None else pd.NaT,
                        "bookmaker_filter": bookmaker_filter, "event_id": ev.get("id"), "game_id": gid,
                        "map_status": why, "commence_time": pd.Timestamp(ev["commence_time"]),
                        "home_team": TEAM_NAMES.get(ev["home_team"]), "away_team": TEAM_NAMES.get(ev["away_team"]),
                        "book": bk["key"], "book_last_update": pd.Timestamp(mk.get("last_update") or bk.get("last_update")),
                        "market": mk["key"], "outcome": oc["name"],
                        "team": TEAM_NAMES.get(oc["name"]) if mk["key"] in ("h2h", "spreads") else None,
                        "point": oc.get("point"), "price": oc.get("price")})
    return pd.DataFrame(rows), unmatched


def normalize_props(ev: dict, pull_id: str, pulled_at, game_id: str | None, season: int,
                    mapper: PlayerMapper, bookmaker_filter: str | None = None, snapshot_ts=None,
                    source: str = "live") -> tuple[pd.DataFrame, list]:
    rows, unmatched = [], []
    teams = (TEAM_NAMES.get(ev.get("home_team")), TEAM_NAMES.get(ev.get("away_team")))
    cache = {}
    for bk in ev.get("bookmakers", []):
        for mk in bk.get("markets", []):
            market = KEY_TO_MARKET.get(mk["key"])
            if market is None:
                continue
            for oc in mk.get("outcomes", []):
                raw = oc.get("description") or ""
                if (raw, market) not in cache:
                    cache[(raw, market)] = mapper.match(raw, game_id, season, teams, market)
                pid, why = cache[(raw, market)]
                if pid is None:
                    unmatched.append({"pull_id": pull_id, "game_id": game_id, "event_id": ev.get("id"),
                                      "book": bk["key"], "market": market, "raw_name": raw, "reason": why})
                rows.append({
                    "pull_id": pull_id, "source": source, "pulled_at": pulled_at,
                    "snapshot_ts": pd.Timestamp(snapshot_ts) if snapshot_ts is not None else pd.NaT,
                    "bookmaker_filter": bookmaker_filter, "event_id": ev.get("id"), "game_id": game_id,
                    "commence_time": pd.Timestamp(ev["commence_time"]) if ev.get("commence_time") else pd.NaT,
                    "book": bk["key"], "book_last_update": pd.Timestamp(mk.get("last_update") or bk.get("last_update")),
                    "market": market, "market_key": mk["key"], "player_raw": raw, "player_id": pid,
                    "side": str(oc["name"]).lower(), "point": oc.get("point"), "price": oc.get("price")})
    df = pd.DataFrame(rows)
    if len(df):
        df = add_prop_devig(df)
    # one unmatched record per (raw name, market)
    um = pd.DataFrame(unmatched).drop_duplicates(["raw_name", "market"]).to_dict("records") if unmatched else []
    return df, um


def add_prop_devig(df: pd.DataFrame) -> pd.DataFrame:
    """De-vig each two-sided prop (over/under or yes/no at the same book, player,
    market, line, pull) with odds.devig (multiplicative). One-sided -> NaN."""
    df = df.copy()
    df["fair_prob"] = np.nan
    key = ["pull_id", "book", "market", "player_raw", "point"]
    pairs = {"over": "under", "under": "over", "yes": "no", "no": "yes"}
    g = df.assign(point=df["point"].fillna(-999)).groupby(key, dropna=False)
    for _, x in g:
        if len(x) != 2 or set(x["side"]) not in ({"over", "under"}, {"yes", "no"}):
            continue
        a, b = x.index
        pa, pb, _ = O.devig(x.loc[a, "price"], x.loc[b, "price"])
        df.loc[a, "fair_prob"], df.loc[b, "fair_prob"] = pa, pb
    return df


def snapshot_time(df: pd.DataFrame) -> pd.Series:
    """When the prices were valid: the API's snapshot time for backfilled rows,
    otherwise the UTC time we pulled them."""
    if "snapshot_ts" not in df:
        return df["pulled_at"]
    return df["snapshot_ts"].where(df["snapshot_ts"].notna(), df["pulled_at"])


def consensus(lines: pd.DataFrame) -> pd.DataFrame:
    """Per (pull, game): median home spread point and total over books, and the
    median of per-book de-vigged home moneyline probabilities (odds.devig)."""
    out = []
    lines = lines[lines["game_id"].notna()]
    for (pid, gid), x in lines.groupby(["pull_id", "game_id"]):
        home = x["home_team"].iloc[0]
        sp = x[(x.market == "spreads") & (x.team == home)]["point"].dropna()
        tot = x[(x.market == "totals") & (x.outcome == "Over")]["point"].dropna()
        ph = []
        for bk, y in x[x.market == "h2h"].groupby("book"):
            hp, ap = y[y.team == home]["price"], y[(y.team != home) & y.team.notna()]["price"]
            if len(hp) == 1 and len(ap) == 1:
                ph.append(O.devig(hp.iloc[0], ap.iloc[0])[0])
        out.append({"pull_id": pid, "game_id": gid, "source": x["source"].iloc[0],
                    "pulled_at": x["pulled_at"].iloc[0],
                    "snapshot_ts": x["snapshot_ts"].iloc[0] if "snapshot_ts" in x else pd.NaT,
                    "bookmaker_filter": x["bookmaker_filter"].iloc[0],
                    "commence_time": x["commence_time"].iloc[0], "home_team": home,
                    "home_spread": float(sp.median()) if len(sp) else np.nan,
                    "total": float(tot.median()) if len(tot) else np.nan,
                    "p_home_ml": float(np.median(ph)) if ph else np.nan,
                    "n_books_spread": int(x[(x.market == "spreads") & (x.team == home)]["book"].nunique()),
                    "n_books_total": int(x[(x.market == "totals")]["book"].nunique()),
                    "n_books_ml": len(ph)})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #
def _append(path: Path, df: pd.DataFrame, key: list) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df = df.drop_duplicates(key, keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def _log_unmatched(path: Path, rows: list):
    if not rows:
        return
    df = pd.DataFrame(rows).assign(logged_at=_iso(utcnow()))
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def _save_raw(sub: str, name: str, payload: dict) -> Path:
    p = ODDS_DIR / sub / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload))
    return p


LINE_KEY = ["pull_id", "event_id", "book", "market", "outcome", "point"]
PROP_KEY = ["pull_id", "event_id", "book", "market", "player_raw", "side", "point"]


def store_lines(df: pd.DataFrame) -> None:
    allv = _append(LINES_PQ, df, LINE_KEY)
    if len(allv):
        consensus(allv).to_parquet(CONSENSUS_PQ, index=False)


def store_props(df: pd.DataFrame) -> None:
    _append(PROPS_PQ, df, PROP_KEY)


# --------------------------------------------------------------------------- #
# Lookups used by the simulator / parlay CLI
# --------------------------------------------------------------------------- #
def _valid_before_kickoff(df: pd.DataFrame, kickoff: pd.Timestamp, as_of: pd.Timestamp | None):
    t = snapshot_time(df)
    m = t < kickoff                                   # never a snapshot from after kickoff
    if "commence_time" in df:
        m &= t < df["commence_time"].fillna(kickoff)
    if as_of is not None:
        m &= t <= as_of
    return df[m], t[m]


def latest_consensus(game_id: str, kickoff: pd.Timestamp, as_of: pd.Timestamp | None = None,
                     path: Path | None = None) -> dict | None:
    path = path or CONSENSUS_PQ
    if not path.exists():
        return None
    c = pd.read_parquet(path)
    c = c[c["game_id"] == game_id]
    if c.empty:
        return None
    c, t = _valid_before_kickoff(c, kickoff, as_of)
    ok = c["home_spread"].notna() & c["total"].notna()
    c, t = c[ok], t[ok]
    if c.empty:
        return None
    r = c.loc[t.idxmax()]
    st = t.max()
    ref = min(kickoff, as_of if as_of is not None else utcnow())
    age_h = (ref - st).total_seconds() / 3600
    return {"home_spread": r["home_spread"], "total": r["total"], "p_home_ml": r["p_home_ml"],
            "n_books": int(min(r["n_books_spread"], r["n_books_total"])), "source": r["source"],
            "snapshot_time": st, "pulled_at": r["pulled_at"], "age_hours": age_h,
            "stale": age_h > STALE_HOURS}


def lookup_price(leg, game_id: str, kickoff: pd.Timestamp, home: str, as_of=None, book: str | None = None,
                 lines_path: Path | None = None, props_path: Path | None = None) -> dict | None:
    """Best (highest-payout) price for a leg in the latest pre-kickoff snapshot of
    each book (or the given book). Exact line match only."""
    if leg.market in PROP_MARKETS:
        path = props_path or PROPS_PQ
        if not path.exists():
            return None
        d = pd.read_parquet(path)
        d = d[(d.game_id == game_id) & (d.player_id == leg.player_id) & (d.market == leg.market)
              & (d.side == leg.side)]
        if leg.line is not None:
            d = d[np.isclose(d["point"].astype(float), float(leg.line))]
    else:
        path = lines_path or LINES_PQ
        if not path.exists():
            return None
        d = pd.read_parquet(path)
        d = d[d.game_id == game_id]
        if leg.market == "moneyline":
            d = d[(d.market == "h2h") & (d.team == leg.side)]
        elif leg.market == "spread":
            d = d[(d.market == "spreads") & (d.team == leg.side) & np.isclose(d["point"].astype(float), float(leg.line))]
        else:
            d = d[(d.market == "totals") & (d.outcome.str.lower() == leg.side) &
                  np.isclose(d["point"].astype(float), float(leg.line))]
    if book:
        d = d[d.book == book]
    if d.empty:
        return None
    d, t = _valid_before_kickoff(d, kickoff, as_of)
    if d.empty:
        return None
    d = d.assign(_t=t)
    d = d.sort_values("_t").groupby("book").tail(1)          # latest snapshot per book
    d = d.assign(_dec=O.american_to_decimal(d["price"].astype(float)))
    r = d.loc[d["_dec"].idxmax()]
    return {"book": r["book"], "price": float(r["price"]), "snapshot_time": r["_t"],
            "pulled_at": r["pulled_at"], "source": r["source"], "n_books_quoting": int(len(d))}


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cost_lines() -> int:
    return len(GAME_MARKETS) * 1          # markets x regions (1 region: us)


def cost_props_event() -> int:
    return len(PROP_MARKETS) * 1


def _book_params(bookmaker: str | None) -> dict:
    """regions=us, or bookmakers=<key> (docs: bookmakers takes priority over regions;
    up to 10 bookmakers = 1 region, so the cost is the same as the us region)."""
    if bookmaker is None:
        return {"regions": REGIONS}
    if bookmaker not in KNOWN_BOOKMAKERS:
        raise ValueError(f"unknown bookmaker {bookmaker!r}; US keys per the docs: "
                         f"{', '.join(sorted(KNOWN_BOOKMAKERS))}")
    return {"bookmakers": bookmaker}


def pull_lines(client: Client | None = None, dry_run: bool = False, sched=None,
               bookmaker: str | None = None) -> dict:
    est = cost_lines()
    bp = _book_params(bookmaker)
    plan = {"calls": [{"endpoint": f"/v4/sports/{SPORT}/odds", "markets": GAME_MARKETS, **bp, "est": est}],
            "est_total": est}
    if dry_run:
        return {"dry_run": True, **plan}
    client = client or Client()
    sched = sched if sched is not None else load_schedule()
    pulled = utcnow()
    data, _ = client.get(f"/v4/sports/{SPORT}/odds",
                         {**bp, "markets": ",".join(GAME_MARKETS),
                          "oddsFormat": "american", "dateFormat": "iso"}, "lines", est, "live game lines")
    tag = f"{_stamp(pulled)}" + (f"_{bookmaker}" if bookmaker else "")
    pid = f"live_lines_{tag}"
    _save_raw("lines", f"{tag}.json", {"pulled_at": _iso(pulled), "bookmaker_filter": bookmaker, "data": data})
    df, um = normalize_lines(data, pid, pulled, sched, bookmaker)
    store_lines(df)
    _log_unmatched(UNMATCHED_GAMES, um)
    return {"pull_id": pid, "events": len(data), "rows": len(df), "unmatched_games": um,
            "books": int(df["book"].nunique()) if len(df) else 0, "est_total": est}


def upcoming_games(sched: pd.DataFrame, now: pd.Timestamp, days: int) -> pd.DataFrame:
    return sched[(sched["kickoff_utc"] > now) & (sched["kickoff_utc"] <= now + pd.Timedelta(days=days))]


def pull_props(client: Client | None = None, dry_run: bool = False, days: int = 7,
               game_id: str | None = None, sched=None, mapper: PlayerMapper | None = None,
               bookmaker: str | None = None) -> dict:
    sched = sched if sched is not None else load_schedule()
    now = utcnow()
    games = upcoming_games(sched, now, days)
    if game_id:
        games = games[games.game_id == game_id]
    est_each = cost_props_event()
    bp = _book_params(bookmaker)
    plan = {"bookmaker_filter": bookmaker,"calls": [{"endpoint": f"/v4/sports/{SPORT}/events (free)", "est": 0}] +
            [{"endpoint": f"/v4/sports/{SPORT}/events/<id>/odds", "game_id": g, "est": est_each}
             for g in games.game_id],
            "est_total": est_each * len(games), "n_games": len(games)}
    if dry_run:
        return {"dry_run": True, **plan}
    client = client or Client()
    mapper = mapper or PlayerMapper()
    events, _ = client.get(f"/v4/sports/{SPORT}/events",
                           {"dateFormat": "iso", "commenceTimeFrom": _iso(now),
                            "commenceTimeTo": _iso(now + pd.Timedelta(days=days))}, "free", 0, "event list")
    res = {"games": [], "unmatched_games": [], "unmatched_players": [], "stopped": None}
    for ev in events:
        gid, why = map_game(ev["home_team"], ev["away_team"], ev["commence_time"], sched)
        if gid is None:
            res["unmatched_games"].append({"event_id": ev["id"], "home": ev["home_team"],
                                           "away": ev["away_team"], "commence_time": ev["commence_time"],
                                           "reason": why})
            continue
        if game_id and gid != game_id:
            continue
        try:
            client.budget.check("props", est_each)
        except BudgetRefused as e:
            res["stopped"] = str(e)
            break
        pulled = utcnow()
        data, hdr = client.get(f"/v4/sports/{SPORT}/events/{ev['id']}/odds",
                               {**bp, "markets": ",".join(PROP_MARKETS.values()),
                                "oddsFormat": "american", "dateFormat": "iso"}, "props", est_each, gid)
        tag = f"{gid}_{_stamp(pulled)}" + (f"_{bookmaker}" if bookmaker else "")
        pid = f"live_props_{tag}"
        _save_raw("props", f"{tag}.json", {"pulled_at": _iso(pulled), "bookmaker_filter": bookmaker, "data": data})
        season = int(sched.loc[sched.game_id == gid, "season"].iloc[0])
        df, um = normalize_props(data, pid, pulled, gid, season, mapper, bookmaker)
        store_props(df)
        res["unmatched_players"] += um
        res["games"].append({"game_id": gid, "rows": len(df), "books": int(df["book"].nunique()) if len(df) else 0,
                             "players": int(df["player_raw"].nunique()) if len(df) else 0,
                             "matched_players": int(df.loc[df.player_id.notna(), "player_raw"].nunique()) if len(df) else 0,
                             "actual_credits": hdr.get("x-requests-last")})
    _log_unmatched(UNMATCHED_GAMES, res["unmatched_games"])
    _log_unmatched(UNMATCHED_PLAYERS, res["unmatched_players"])
    return res


# --------------------------------------------------------------------------- #
# One-time historical backfill (approved: 2026 season to date)
# --------------------------------------------------------------------------- #
# 10 US books in one request (docs: up to 10 bookmakers = 1 region), incl. fanatics.
# williamhill_us was swapped for betparx after check-books: at the 2026 opener
# williamhill_us offered 1 of our 9 prop markets, betparx 9 of 9.
BACKFILL_BOOKS = ["draftkings", "fanduel", "betmgm", "betparx", "fanatics",
                  "betrivers", "espnbet", "hardrockbet", "bovada", "betonlineag"]
BACKFILL_SEASONS = (2026,)        # the only approved season
SNAPSHOT_LEAD = pd.Timedelta(minutes=1)   # request date = kickoff - 1 min -> closest snapshot before kickoff
EMPTY_STEPS_BACK = 6              # if a props snapshot is empty (billed 0), step back up to 6 x 5 min


def cost_hist_lines() -> int:
    return 10 * len(GAME_MARKETS) * 1      # 10 x markets x regions (10 books = 1 region)


def cost_hist_props_event() -> int:
    return 10 * len(PROP_MARKETS) * 1      # upper bound: 10 x unique markets returned x regions


def _load_state(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"slots": {}, "games": {}}


def _save_state(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, default=str))
    tmp.replace(path)


def backfill_games(sched: pd.DataFrame, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Approved scope: 2026 games completed before today (US Eastern date)."""
    today = (today or utcnow()).tz_convert(ET).normalize().tz_localize(None)
    g = sched[sched.season.isin(BACKFILL_SEASONS) & sched.label_home_score.notna()
              & (sched.gameday < today) & sched.kickoff_utc.notna()]
    return g.sort_values(["kickoff_utc", "game_id"])


def backfill_plan(sched=None, state=None, today=None) -> list[dict]:
    """Historical calls still needed: per kickoff slot one game-lines snapshot,
    per game one props snapshot, both at (kickoff - 1 min). Saved items are skipped."""
    sched = sched if sched is not None else load_schedule()
    state = state if state is not None else _load_state(BACKFILL_STATE)
    plan = []
    for ko, slot in backfill_games(sched, today).groupby("kickoff_utc", sort=True):
        skey = _iso(ko)
        at = _iso(ko - SNAPSHOT_LEAD)
        if skey not in state["slots"]:
            plan.append({"kind": "lines", "slot": skey, "date": at, "game_ids": list(slot.game_id),
                         "est": cost_hist_lines()})
        for gid in slot.game_id:
            if state["games"].get(gid, {}).get("props") is None:
                plan.append({"kind": "props", "slot": skey, "date": at, "game_id": gid,
                             "est": cost_hist_props_event()})
    return plan


def _hist_params(date: str, markets: list) -> dict:
    return {"bookmakers": ",".join(BACKFILL_BOOKS), "markets": ",".join(markets),
            "oddsFormat": "american", "dateFormat": "iso", "date": date}


def backfill(client: Client | None = None, dry_run: bool = False, confirm: bool = False,
             sched=None, mapper: PlayerMapper | None = None, state_path: Path | None = None,
             max_calls: int | None = None, today=None) -> dict:
    state_path = state_path or BACKFILL_STATE
    sched = sched if sched is not None else load_schedule()
    state = _load_state(state_path)
    plan = backfill_plan(sched, state, today)
    games = backfill_games(sched, today)
    summary = {"seasons": list(BACKFILL_SEASONS), "books": BACKFILL_BOOKS,
               "games_in_scope": int(len(games)), "kickoff_slots": int(games.kickoff_utc.nunique()),
               "first_game": games.game_id.iloc[0] if len(games) else None,
               "last_game": games.game_id.iloc[-1] if len(games) else None,
               "calls_remaining": len(plan),
               "est_lines": int(sum(p["est"] for p in plan if p["kind"] == "lines")),
               "est_props": int(sum(p["est"] for p in plan if p["kind"] == "props")),
               "est_total": int(sum(p["est"] for p in plan))}
    if dry_run:
        return {"dry_run": True, **summary}
    if not confirm:
        raise BudgetRefused("backfill spends real credits: run --dry-run first, then re-run with --confirm "
                            "once the cost is approved")
    client = client or Client()
    mapper = mapper or PlayerMapper()
    out = {**summary, "done": 0, "stopped": None, "billed": 0, "unmatched_games": [], "unmatched_players": []}

    def call(path, params, est, note):
        try:
            d, h = client.get(path, params, "historical", est, note, allow_historical=True)
        except CostOverrun as e:
            out["stopped"] = str(e)
            d, h = e.data, e.headers
        out["billed"] += int(float(h.get("x-requests-last") or 0))
        return d

    for i, item in enumerate(plan):
        if out["stopped"]:
            break
        if max_calls is not None and i >= max_calls:
            out["stopped"] = f"max_calls={max_calls} reached"
            break
        try:
            client.budget.check("historical", item["est"])
        except BudgetRefused as e:
            out["stopped"] = f"{e}; resume later from {item['kind']} {item.get('game_id', item['slot'])}"
            break
        if item["kind"] == "lines":
            raw_p = ODDS_DIR / "historical_lines" / f"{item['slot'].replace(':', '')}.json"
            if raw_p.exists():                      # already paid for: never pay twice
                payload = json.loads(raw_p.read_text())
            else:
                payload = call(f"/v4/historical/sports/{SPORT}/odds", _hist_params(item["date"], GAME_MARKETS),
                               item["est"], f"slot {item['slot']}")
                _save_raw("historical_lines", raw_p.name, payload)
            ts = payload.get("timestamp")
            evs, ev_ids = [], {}
            for e in payload.get("data") or []:   # snapshot lists all upcoming games; keep this slot's
                gid, _ = map_game(e["home_team"], e["away_team"], e["commence_time"], sched)
                if gid in item["game_ids"]:
                    evs.append(e)
                    ev_ids[gid] = e["id"]
            out["unmatched_games"] += [{"slot": item["slot"], "game_id": g_, "reason": "not in historical snapshot"}
                                       for g_ in item["game_ids"] if g_ not in ev_ids]
            df, um = normalize_lines(evs, f"hist_lines_{item['slot'].replace(':', '')}", utcnow(), sched,
                                     ",".join(BACKFILL_BOOKS), ts, "historical")
            store_lines(df)
            out["unmatched_games"] += um
            state["slots"][item["slot"]] = {"snapshot_ts": ts, "events": ev_ids}
        else:
            gid = item["game_id"]
            ev_id = state["slots"].get(item["slot"], {}).get("events", {}).get(gid)
            if ev_id is None:
                state["games"].setdefault(gid, {})["props"] = "no_event_id"
                _save_state(state, state_path)
                continue
            raw_p = ODDS_DIR / "historical_props" / f"{gid}.json"
            if raw_p.exists():
                payload = json.loads(raw_p.read_text())
            else:
                date = item["date"]
                for step in range(EMPTY_STEPS_BACK + 1):
                    payload = call(f"/v4/historical/sports/{SPORT}/events/{ev_id}/odds",
                                   _hist_params(date, list(PROP_MARKETS.values())), item["est"], gid)
                    ev = payload.get("data") or {}
                    if ev.get("bookmakers") or out["stopped"] or not payload.get("previous_timestamp"):
                        break
                    date = payload["previous_timestamp"]      # empty responses are not billed
                _save_raw("historical_props", raw_p.name, payload)
            ev = payload.get("data") or {}
            ts = payload.get("timestamp")
            if ev.get("bookmakers"):
                df, um = normalize_props(ev, f"hist_props_{gid}", utcnow(), gid, int(item.get("season", 2026)),
                                         mapper, ",".join(BACKFILL_BOOKS), ts, "historical")
                store_props(df)
                out["unmatched_players"] += um
            state["games"].setdefault(gid, {})["props"] = ts if ev.get("bookmakers") else "empty"
        _save_state(state, state_path)
        out["done"] += 1
    _log_unmatched(UNMATCHED_GAMES, out["unmatched_games"])
    _log_unmatched(UNMATCHED_PLAYERS, out["unmatched_players"])
    out["calls_remaining"] = len(backfill_plan(sched, _load_state(state_path), today))
    if out["calls_remaining"] == 0:
        client.budget.consume_approval()     # one-time: the historical endpoint is disabled again
        out["approval"] = "consumed: historical endpoint disabled until a new explicit approval"
    return out


def check_books(client: Client | None = None, dry_run: bool = False, confirm: bool = False,
                sched=None, today=None) -> dict:
    """Which of BACKFILL_BOOKS had which markets at the start of the backfill window:
    historical events (1 credit) + historical event markets (1 credit) for the
    first game in scope. The docs do not publish per-book start dates."""
    sched = sched if sched is not None else load_schedule()
    g = backfill_games(sched, today)
    first = g.iloc[0]
    date = _iso(first.kickoff_utc - SNAPSHOT_LEAD)
    plan = {"game_id": first.game_id, "date": date, "est_total": 2,
            "calls": ["historical events (1)", "historical event markets (1)"]}
    if dry_run:
        return {"dry_run": True, **plan}
    if not confirm:
        raise BudgetRefused("check-books spends 2 historical credits: re-run with --confirm once approved")
    client = client or Client()
    evs, _ = client.get(f"/v4/historical/sports/{SPORT}/events", {"date": date, "dateFormat": "iso"},
                        "historical", 1, "check-books events", allow_historical=True)
    ev_id = None
    for e in evs.get("data") or []:
        gid, _ = map_game(e["home_team"], e["away_team"], e["commence_time"], sched)
        if gid == first.game_id:
            ev_id = e["id"]
    if ev_id is None:
        return {**plan, "error": "first game not found in historical events"}
    mk, _ = client.get(f"/v4/historical/sports/{SPORT}/events/{ev_id}/markets",
                       {"date": date, "dateFormat": "iso", "bookmakers": ",".join(BACKFILL_BOOKS)},
                       "historical", 1, "check-books markets", allow_historical=True)
    books = {b["key"]: sorted(m["key"] for m in b.get("markets", [])) for b in (mk.get("data") or {}).get("bookmakers", [])}
    return {**plan, "snapshot_ts": mk.get("timestamp"),
            "books": {b: {"present": b in books,
                          "prop_markets": [k for k in books.get(b, []) if k in KEY_TO_MARKET],
                          "game_markets": [k for k in books.get(b, []) if k in GAME_MARKETS]}
                      for b in BACKFILL_BOOKS}}


def credits_report(budget: Budget | None = None) -> dict:
    b = budget or Budget.from_env()
    month = utcnow().strftime("%Y-%m")
    rep = {"month": month, "remaining_per_api": b.remaining(), "reserve": b.reserve, "by_category": {}}
    for c in ("lines", "props", "historical"):
        rep["by_category"][c] = {"used": b.used(c, month), "cap": b.cap(c)}
    rep["historical_approval"] = {"approval": b.historical_approval, "active_cap": b.historical_cap(),
                                  "used_up": bool(b.historical_approval) and b.historical_approval in b.approvals_used()}
    calls = [e for e in b.entries() if e.get("month") == month and not e.get("dry_run")]
    rep["calls_this_month"] = len(calls)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m nflsim.odds_feed")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pull-lines", "pull-props", "latest", "credits", "backfill", "check-books"):
        p = sub.add_parser(name)
        p.add_argument("--dry-run", action="store_true")
        if name in ("pull-lines", "pull-props"):
            p.add_argument("--bookmaker", help="limit to one book, e.g. fanatics (same credit cost as "
                           "the us region: up to 10 bookmakers = 1 region)")
        if name == "pull-props":
            p.add_argument("--days", type=int, default=7)
            p.add_argument("--game-id")
        if name in ("backfill", "check-books"):
            p.add_argument("--confirm", action="store_true",
                           help="actually spend historical credits (after an approved --dry-run)")
        if name == "backfill":
            p.add_argument("--max-calls", type=int)
        if name == "latest":
            p.add_argument("--game-id")
        if name == "credits":
            p.add_argument("--refresh", action="store_true",
                           help="free /v4/sports call to read remaining credits")
    a = ap.parse_args(argv)
    if a.cmd == "pull-lines":
        r = pull_lines(dry_run=a.dry_run, bookmaker=a.bookmaker)
    elif a.cmd == "pull-props":
        r = pull_props(dry_run=a.dry_run, days=a.days, game_id=a.game_id, bookmaker=a.bookmaker)
    elif a.cmd == "backfill":
        r = backfill(dry_run=a.dry_run, confirm=a.confirm, max_calls=a.max_calls)
    elif a.cmd == "check-books":
        r = check_books(dry_run=a.dry_run, confirm=a.confirm)
    elif a.cmd == "latest":
        r = latest_cmd(a.game_id)
    else:
        if a.refresh and not a.dry_run:
            Client().get("/v4/sports", {}, "free", 0, "refresh remaining credits")
        r = credits_report()
        if a.dry_run:
            r["dry_run"] = True
    print(json.dumps(r, indent=1, default=str))


def latest_cmd(game_id: str | None) -> dict:
    sched = load_schedule()
    now = utcnow()
    games = sched[sched.game_id == game_id] if game_id else upcoming_games(sched, now, 8)
    out = []
    props = pd.read_parquet(PROPS_PQ) if PROPS_PQ.exists() else pd.DataFrame()
    for g in games.itertuples():
        c = latest_consensus(g.game_id, g.kickoff_utc, as_of=min(now, g.kickoff_utc))
        rec = {"game_id": g.game_id, "kickoff_utc": _iso(g.kickoff_utc), "consensus": c}
        if len(props):
            p = props[props.game_id == g.game_id]
            if len(p):
                p, t = _valid_before_kickoff(p, g.kickoff_utc, now)
                if len(p):
                    last = p.assign(_t=t).sort_values("_t").groupby("book").tail(1)["pull_id"]
                    p = p[p.pull_id.isin(set(last))]
                    rec["props"] = {"rows": len(p), "books": int(p.book.nunique()),
                                    "players": int(p.player_raw.nunique()),
                                    "matched_players": int(p.loc[p.player_id.notna(), "player_raw"].nunique()),
                                    "latest_snapshot": _iso(t.max())}
        out.append(rec)
    return {"games": out}


if __name__ == "__main__":
    main(sys.argv[1:])
