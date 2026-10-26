"""Logic behind the Streamlit market browser and parlay builder (Milestone 3).

Kept out of app/streamlit_app.py so it can be tested. Everything here reads
saved files only: odds snapshots (odds_feed), nflverse results, and the user's
parlay log. Nothing here calls The Odds API.

Market board: the latest pre-kickoff snapshot of one book for one game, grouped
like a sportsbook (game lines first, then by player). Every offering is either
priceable by the simulator (a markets.Leg can be built) or greyed out with the
reason. Nothing that is not in a saved snapshot is shown, and nothing greyed out
is ever priced.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import odds as O
from .ingest import PROCESSED
from .markets import LOSS, PLAYER_MARKETS, PUSH, VOID, WIN, Leg

LOG_PATH = PROCESSED / "parlay_log.jsonl"
DEFAULT_BOOK = "fanatics"
GAME_MARKET_NAMES = {"h2h": "Moneyline", "spreads": "Spread", "totals": "Total"}
QB_MARKETS = {"passing_yards", "passing_tds", "pass_attempts", "completions"}
RECEIVER_MARKETS = {"receptions", "receiving_yards", "anytime_td"}

# Alternate prop lines: odds_feed.ALT_PROP_MARKETS (official docs keys), pulled
# with `odds_feed pull-props --alternates` (costs credits: needs approval).
# Alternate spreads/totals (alternate_spreads, alternate_totals) are not supported
# by the feed yet; if they ever appear in a snapshot they are shown and priced.
ALT_GAME_KEYS = ["alternate_spreads", "alternate_totals"]


def alt_cost_per_game(regions: int = 1) -> int:
    """Extra credits (upper bound) per game for the alternate prop markets."""
    from .odds_feed import ALT_PROP_MARKETS
    return len(ALT_PROP_MARKETS) * regions


def _snap_time(df: pd.DataFrame) -> pd.Series:
    return df["snapshot_ts"].fillna(df["pulled_at"])


def _latest_pre_kickoff(df: pd.DataFrame, book: str, as_of: pd.Timestamp | None,
                       market_col: str = "market") -> pd.DataFrame:
    """The book's latest pre-kickoff snapshot of each market (a pull of a few
    markets must not hide the other markets from an earlier pull)."""
    d = df[df["book"] == book].copy()
    if d.empty:
        return d
    d["t"] = _snap_time(d)
    d = d[d["t"] < d["commence_time"]]          # never a post-kickoff price
    if as_of is not None:
        d = d[d["t"] <= as_of]
    if d.empty:
        return d
    return d[d["t"] == d.groupby(market_col)["t"].transform("max")]


def load_snapshots(lines_path: Path | None = None, props_path: Path | None = None):
    from .odds_feed import LINES_PQ, PROPS_PQ
    lp, pp = lines_path or LINES_PQ, props_path or PROPS_PQ
    lines = pd.read_parquet(lp) if lp.exists() else pd.DataFrame()
    props = pd.read_parquet(pp) if pp.exists() else pd.DataFrame()
    return lines, props


def game_list(lines: pd.DataFrame, props: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Games with at least one saved snapshot, newest kickoff first."""
    ids = set(lines.get("game_id", pd.Series(dtype=str)).dropna()) | \
        set(props.get("game_id", pd.Series(dtype=str)).dropna())
    g = games[games["game_id"].isin(ids)].copy()
    g["played"] = g["label_home_score"].notna()
    g["label"] = [f"{r.season} wk{r.week}: {r.away_team} @ {r.home_team}"
                  + (f"  (final {int(r.label_away_score)}-{int(r.label_home_score)})" if r.played else "")
                  for r in g.itertuples()]
    return g.sort_values(["gameday", "game_id"], ascending=[False, True]).reset_index(drop=True)


def books_for_game(lines: pd.DataFrame, props: pd.DataFrame, game_id: str) -> list:
    b = set()
    for d in (lines, props):
        if len(d):
            b |= set(d.loc[d["game_id"] == game_id, "book"])
    return sorted(b)


def market_board(game_id: str, book: str, lines: pd.DataFrame, props: pd.DataFrame,
                 home: str, away: str, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per offering in the book's latest pre-kickoff snapshot.
    Columns: group, order, market, label, side, point, price, player_id, priceable,
    reason, alternate, snapshot_time, leg (markets.Leg or None)."""
    rows = []
    if len(lines):
        d = _latest_pre_kickoff(lines[lines["game_id"] == game_id], book, as_of)
        for r in d.itertuples():
            alt = r.market in ALT_GAME_KEYS
            base = {"alternate_spreads": "spreads", "alternate_totals": "totals"}.get(r.market, r.market)
            leg, reason = None, ""
            if base == "h2h":
                leg = Leg("moneyline", r.team)
                label = f"{r.team} ML"
            elif base == "spreads":
                leg = Leg("spread", r.team, float(r.point))
                label = f"{r.team} {float(r.point):+g}"
            elif base == "totals":
                leg = Leg("total", str(r.outcome).lower(), float(r.point))
                label = f"{str(r.outcome)} {float(r.point):g}"
            else:
                label, reason = f"{r.market} {r.outcome}", "market the simulator can't price"
            if leg is not None and r.team not in (home, away, None) and base != "totals":
                leg, reason = None, "team not in this game"
            rows.append({"group": "Game lines", "order": 0,
                         "market": GAME_MARKET_NAMES.get(base, r.market), "label": label,
                         "side": r.outcome, "point": r.point, "price": float(r.price), "player_id": None,
                         "priceable": leg is not None, "reason": reason, "alternate": alt,
                         "snapshot_time": r.t, "leg": leg})
    if len(props):
        from .odds_feed import market_from_key
        d = _latest_pre_kickoff(props[props["game_id"] == game_id], book, as_of, "market_key")
        for r in d.itertuples():
            key = str(r.market_key)
            alt = key.endswith("_alternate")
            mkt = market_from_key(key)
            leg, reason = None, ""
            if mkt not in PLAYER_MARKETS:
                reason = "market the simulator can't price"
            elif not isinstance(r.player_id, str):
                reason = ("not a player (team/defense/no-scorer outcome)"
                          if any(w in str(r.player_raw) for w in ("D/ST", "Defense", "No Scorer"))
                          else "player not matched to a roster (never guessed)")
            elif mkt == "anytime_td":
                if r.side in ("yes", "no"):
                    leg = Leg("anytime_td", r.side, None, r.player_id)
            elif r.side in ("over", "under") and pd.notna(r.point):
                leg = Leg(mkt, r.side, float(r.point), r.player_id)
            if leg is None and not reason:
                reason = "unsupported outcome"
            nice = (mkt or key).replace("_", " ")
            pt = "" if pd.isna(r.point) else f" {float(r.point):g}"
            rows.append({"group": str(r.player_raw), "order": 1, "market": nice,
                         "label": f"{r.player_raw} {nice} {r.side}{pt}", "side": r.side, "point": r.point,
                         "price": float(r.price), "player_id": r.player_id if isinstance(r.player_id, str) else None,
                         "priceable": leg is not None, "reason": reason, "alternate": alt,
                         "snapshot_time": r.t, "leg": leg})
    if not rows:
        return pd.DataFrame(columns=["group", "order", "market", "label", "side", "point", "price",
                                     "player_id", "priceable", "reason", "alternate", "snapshot_time", "leg"])
    b = pd.DataFrame(rows)
    return b.sort_values(["order", "group", "market", "alternate", "point", "side"],
                         na_position="first").reset_index(drop=True)


def qb_receiver_pairs(legs: list, teams: dict) -> list:
    """(QB leg, receiver leg) pairs where a QB passing leg is combined with a leg
    on a WR/TE of the same team. `teams` = PlayerSim.teams (side -> TeamPlayers).
    The simulator understates these correlations (README: Known limitations)."""
    where = {}
    for tp in teams.values():
        for pid, pos in zip(tp.ids, tp.position):
            where[pid] = (tp.team, pos)
    qbs = [l for l in legs if l.player_id and l.market in QB_MARKETS
           and where.get(l.player_id, (None, None))[1] == "QB"]
    recs = [l for l in legs if l.player_id and l.market in RECEIVER_MARKETS
            and where.get(l.player_id, (None, None))[1] in ("WR", "TE")]
    return [(q, r) for q in qbs for r in recs if where[q.player_id][0] == where[r.player_id][0]]


QB_WR_WARNING = ("This parlay pairs a QB with his own WR/TE. Out of sample (2024-25) the model's "
                 "QB-receiver correlations run about 0.04-0.07 below actual, so the joint probability "
                 "of these legs is likely UNDERSTATED (fair odds too long). See README, Known limitations.")


# --------------------------------------------------------------------------- #
# Grading against official results
# --------------------------------------------------------------------------- #
ACTUAL_COLS = {"passing_yards": "passing_yards", "passing_tds": "passing_tds",
               "pass_attempts": "attempts", "completions": "completions",
               "rushing_yards": "rushing_yards", "rush_attempts": "carries",
               "receptions": "receptions", "receiving_yards": "receiving_yards"}
TD_COLS = ("rushing_tds", "receiving_tds", "special_teams_tds", "fumble_recovery_tds")


def grade_leg_actual(leg: Leg, home: str, away: str, home_score: float, away_score: float,
                     stat_row: dict | None, played: bool) -> int:
    """WIN / LOSS / PUSH / VOID on official results. Player legs are VOID if the
    player did not play (books void them). Anytime TD counts any non-passing TD."""
    if leg.market == "total":
        t = home_score + away_score
        r = WIN if t > leg.line else LOSS if t < leg.line else PUSH
        return r if leg.side == "over" or r == PUSH else (LOSS if r == WIN else WIN)
    if leg.market in ("moneyline", "spread"):
        m = (home_score - away_score) if leg.side == home else (away_score - home_score)
        if leg.market == "spread":
            m = m + leg.line
        return WIN if m > 0 else LOSS if m < 0 else PUSH
    if not played:
        return VOID
    s = stat_row or {}
    if leg.market == "anytime_td":
        tds = sum(float(s.get(c, 0) or 0) for c in TD_COLS)
        return WIN if (tds >= 1) == (leg.side == "yes") else LOSS
    x = float(s.get(ACTUAL_COLS[leg.market], 0) or 0)
    r = WIN if x > leg.line else LOSS if x < leg.line else PUSH
    return r if leg.side == "over" or r == PUSH else (LOSS if r == WIN else WIN)


def actual_results(game_id: str):
    """(game row, {player_id: stat dict}, set of players who played) or None if
    the game has no final score yet. Needs an up-to-date nflverse ingest."""
    from .features import read_processed
    from .ingest import load
    games = read_processed("games")
    g = games[games["game_id"] == game_id]
    if g.empty or pd.isna(g.iloc[0]["label_home_score"]):
        return None
    g = g.iloc[0]
    try:
        st = load("player_stats", [int(g["season"])]).to_pandas()
    except Exception:
        return None
    st = st[st["game_id"] == game_id]
    if st.empty:
        return None
    pg = pd.read_parquet(PROCESSED / "player_games.parquet", columns=["game_id", "player_id"])
    played = set(pg.loc[pg["game_id"] == game_id, "player_id"]) | set(st["player_id"])
    return g, {r["player_id"]: r for r in st.to_dict("records")}, played


def grade_parlay_actual(legs: list, game_id: str, leg_odds: list | None = None,
                        parlay_odds: float | None = None, settlement: str = "reduce") -> dict | None:
    """Grade each leg and settle the parlay on official results; None if not final."""
    from .parlay import settle
    res = actual_results(game_id)
    if res is None:
        return None
    g, stats, played = res
    G = np.array([[grade_leg_actual(l, g["home_team"], g["away_team"], g["label_home_score"],
                                    g["label_away_score"], stats.get(l.player_id), l.player_id in played)
                   for l in legs]], dtype=np.int8)
    dec = np.asarray(O.american_to_decimal(leg_odds), float) if leg_odds else None
    pdec = float(O.american_to_decimal(parlay_odds)) if parlay_odds is not None else None
    s = settle(G, settlement, dec, pdec)
    payout = float(s["payout"][0])
    names = {WIN: "win", LOSS: "loss", PUSH: "push", VOID: "void"}
    status = ("win" if s["p_full_win"] else "reduced win" if s["p_reduced_win"] else
              "refund" if s["p_refund"] else "loss")
    return {"legs": [names[int(x)] for x in G[0]], "status": status,
            "payout_per_unit": None if np.isnan(payout) else payout,
            "final": f"{g['away_team']} {int(g['label_away_score'])} - {g['home_team']} {int(g['label_home_score'])}"}


# --------------------------------------------------------------------------- #
# Parlay log (local JSON lines, git-ignored data folder)
# --------------------------------------------------------------------------- #
def leg_to_dict(l: Leg) -> dict:
    return {"market": l.market, "side": l.side, "line": l.line, "player_id": l.player_id, "label": l.label}


def leg_from_dict(d: dict) -> Leg:
    return Leg(d["market"], d["side"], d.get("line"), d.get("player_id"), d.get("label"))


def log_parlay(entry: dict, path: Path | None = None) -> None:
    path = path or LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def load_log(path: Path | None = None) -> list:
    path = path or LOG_PATH
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def grade_log(path: Path | None = None) -> list:
    """Grade every logged parlay whose game is final; rewrite the log with results."""
    path = path or LOG_PATH
    out = []
    for e in load_log(path):
        if e.get("result") is None:
            legs = [leg_from_dict(d) for d in e["legs"]]
            e["result"] = grade_parlay_actual(legs, e["game_id"], e.get("leg_odds"),
                                              e.get("parlay_odds"), e.get("settlement", "reduce"))
        out.append(e)
    if out:
        path.write_text("".join(json.dumps(e, default=str) + "\n" for e in out))
    return out
