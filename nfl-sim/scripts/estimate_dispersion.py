"""Estimate the Dirichlet concentration of per-game target / carry shares.

Uses 2023 only (before the 2024-25 player validation seasons). For every team-game,
the as-of model shares are re-normalized over the players who actually played
(this is an estimate of game-to-game share variation, so conditioning on the
realized active set is intended), then the Dirichlet-multinomial variance
    Var(x) = n s (1 - s) (n + alpha) / (1 + alpha)
is matched by method of moments: alpha = (C - A) / (A - B) with
A = sum (x - n s)^2, B = sum n s (1 - s), C = sum n^2 s (1 - s).
The excess variance includes share-model error, which is intended: it widens
predictive distributions by the amount the model actually misses.
"""
import sys

import numpy as np
import pandas as pd

from nflsim.pipeline import load_context
from nflsim.players import fit_team_players

season = int(sys.argv[1]) if len(sys.argv) > 1 else 2023
ctx = load_context()
g = ctx.games[(ctx.games.season == season) & ctx.games.label_home_score.notna()]
gf = ctx.gf.set_index("game_id")
pg = ctx.player_games.set_index(["game_id", "player_id"])
acc = {"tgt": [0.0, 0.0, 0.0, 0], "car": [0.0, 0.0, 0.0, 0]}
for gm in g.itertuples():
    pri = ctx.priors(gm.gameday, season)
    for side, t in (("home", gm.home_team), ("away", gm.away_team)):
        tp = fit_team_players(gm.game_id, t, gf.loc[gm.game_id, side + "_qb_id"], ctx.player_games,
                              ctx.roles, pri, gm.gameday, season, hist_by_player=ctx.hist_by_player)
        rows = [pg.loc[(gm.game_id, p)] if (gm.game_id, p) in pg.index else None for p in tp.ids]
        played = np.array([r is not None for r in rows])
        if not played.any():
            continue
        for key, share, cnt, tot in (("tgt", tp.tgt_share, "targets", "team_targets"),
                                     ("car", tp.car_share, "carries", "team_carries")):
            s = share * played
            if s.sum() <= 0:
                continue
            s = s / s.sum()
            n = next(float(r[tot]) for r in rows if r is not None)
            x = np.array([float(r[cnt]) if r is not None else 0.0 for r in rows])
            m = (s > 0) & (s < 1)
            a = acc[key]
            a[0] += ((x[m] - n * s[m]) ** 2).sum()
            a[1] += (n * s[m] * (1 - s[m])).sum()
            a[2] += (n * n * s[m] * (1 - s[m])).sum()
            a[3] += int(m.sum())
for key, (A, B, C, k) in acc.items():
    alpha = (C - A) / (A - B) if B < A < C else float("nan")
    print(f"{key}: player-games={k}  A={A:.0f} B={B:.0f} C={C:.0f}  alpha={alpha:.2f}  "
          f"(variance inflation at n=35: {(35 + alpha) / (1 + alpha):.2f}x binomial)")
