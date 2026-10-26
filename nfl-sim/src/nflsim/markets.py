"""Bet leg definitions and grading against simulated games.

Each leg is graded in every simulation as WIN, LOSS, PUSH or VOID:
  * PUSH: the stat lands exactly on the line (whole-number lines), or a
    moneyline game ends tied.
  * VOID: a player leg whose player does not play. A player not on the pregame
    roster (or listed Out/Doubtful) is void in every simulation; other players
    sit out a simulation with probability 1 - P(active), estimated from their
    recent activity (players.py), and their legs are void in those simulations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LOSS, WIN, PUSH, VOID = 0, 1, 2, 3

PLAYER_MARKETS = {
    "passing_yards": "pass_yards",
    "passing_tds": "pass_td",
    "pass_attempts": "pass_att",
    "completions": "completions",
    "rushing_yards": "rush_yards",
    "rush_attempts": "rush_att",
    "receptions": "receptions",
    "receiving_yards": "rec_yards",
    "anytime_td": None,  # rush_td + rec_td >= 1
}
GAME_MARKETS = ("spread", "total", "moneyline")


@dataclass(frozen=True)
class Leg:
    market: str
    side: str                  # over/under, yes/no (anytime TD), or team abbr (spread/moneyline)
    line: float | None = None  # stat line, team spread (e.g. -3.5), or total
    player_id: str | None = None
    label: str | None = None

    def __post_init__(self):
        if self.market in PLAYER_MARKETS:
            if not self.player_id:
                raise ValueError(f"{self.market} needs player_id")
            if self.market == "anytime_td":
                if self.side not in ("yes", "no"):
                    raise ValueError("anytime_td side must be yes/no")
            elif self.side not in ("over", "under") or self.line is None:
                raise ValueError(f"{self.market} needs side over/under and a line")
        elif self.market == "total":
            if self.side not in ("over", "under") or self.line is None:
                raise ValueError("total needs over/under and a line")
        elif self.market == "spread":
            if self.line is None:
                raise ValueError("spread needs a line")
        elif self.market != "moneyline":
            raise ValueError(f"unknown market {self.market}")

    def name(self) -> str:
        if self.label:
            return self.label
        who = self.player_id or ""
        ln = "" if self.line is None else f" {self.line:+g}" if self.market == "spread" else f" {self.line:g}"
        return f"{who} {self.market} {self.side}{ln}".strip()


def _over_under(x: np.ndarray, line: float, side: str) -> np.ndarray:
    g = np.where(x > line, WIN, np.where(x < line, LOSS, PUSH))
    if side == "under":
        g = np.where(g == WIN, LOSS, np.where(g == LOSS, WIN, PUSH))
    return g.astype(np.int8)


def grade(leg: Leg, game) -> np.ndarray:
    """Grade a leg in every simulation. `game` is a pipeline.SimulatedGame."""
    n = game.n_sims
    if leg.market in GAME_MARKETS:
        home, away = game.home_team, game.away_team
        hs, as_ = game.sim.home_score, game.sim.away_score
        if leg.market == "total":
            return _over_under(hs + as_, leg.line, leg.side)
        if leg.side not in (home, away):
            raise ValueError(f"team {leg.side} not in game {away}@{home}")
        margin = (hs - as_) if leg.side == home else (as_ - hs)
        if leg.market == "moneyline":
            return np.where(margin > 0, WIN, np.where(margin < 0, LOSS, PUSH)).astype(np.int8)
        adj = margin + leg.line
        return np.where(adj > 0, WIN, np.where(adj < 0, LOSS, PUSH)).astype(np.int8)
    ps = game.players
    active = ps.player_active(leg.player_id)
    if active is None:
        return np.full(n, VOID, np.int8)
    if leg.market == "anytime_td":
        tds = ps.player_stat(leg.player_id, "rush_td") + ps.player_stat(leg.player_id, "rec_td")
        g = np.where((tds >= 1) == (leg.side == "yes"), WIN, LOSS).astype(np.int8)
    else:
        g = _over_under(ps.player_stat(leg.player_id, PLAYER_MARKETS[leg.market]), leg.line, leg.side)
    return np.where(active, g, VOID).astype(np.int8)
