"""Replay the anchored player walk-forward's seeds (2024-25, 1,000 sims) and check
that every game's simulated mean margin/total hit the anchors within tolerance."""
import numpy as np
import pandas as pd

from nflsim.anchor import TOL
from nflsim.pipeline import load_context, simulate_game
from nflsim.team_model import fit_team_model

ctx = load_context()
g = ctx.games
test = g[g.season.isin([2024, 2025]) & g.label_home_score.notna()]
rng = np.random.default_rng(0)      # same seed sequence as backtest.player_walk_forward
rows = []
for (s, w), wk in test.groupby(["season", "week"], sort=True):
    cutoff = wk.gameday.min()
    tm = fit_team_model(cutoff, ctx.drives, ctx.pat, ctx.gf)
    for r in wk.itertuples():
        sg = simulate_game(ctx, r.game_id, 1000, seed=int(rng.integers(1 << 31)), cutoff=cutoff,
                           team_model=tm, mode="market_anchored")
        a = sg.anchor
        rows.append({"game_id": r.game_id, "mode": a["mode"],
                     "err_m": sg.sim.margin.mean() - a.get("target_margin", np.nan),
                     "err_t": sg.sim.total.mean() - a.get("target_total", np.nan)})
d = pd.DataFrame(rows)
bad = d[(d.err_m.abs() > TOL) | (d.err_t.abs() > TOL)]
print(f"games {len(d)}, anchored {int((d['mode'] == 'market_anchored').sum())}, outside ±{TOL}: {len(bad)}, "
      f"max |margin err| {d.err_m.abs().max():.3f}, max |total err| {d.err_t.abs().max():.3f}")
if len(bad):
    print(bad.to_string(index=False))
