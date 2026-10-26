"""Joint evaluation of parlay legs on shared simulated games.

The parlay probability is K/N: the number of simulations in which every leg
won, divided by the number of simulations. Leg probabilities are never
multiplied together, except to show the independence benchmark for comparison.

Two kinds of uncertainty are reported separately:
  * Monte Carlo error: from a finite number of simulations. It shrinks as N grows.
  * Model error: from the fitted models. Estimated by refitting the drive model
    and player usage models on bootstrap resamples and re-running the parlay.
    More simulations do NOT reduce it.

Settlement of pushed / void legs (configurable):
  "reduce" (default): pushed or void legs are removed and the parlay pays at the
      reduced odds (the product of the remaining legs' decimal odds; needs
      per-leg offered odds). If every leg pushes or voids, the stake is refunded.
  "void_parlay": any pushed or void leg refunds the whole parlay.
  "push_loses": a pushed or void leg counts as a loss.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import odds as O
from .markets import LOSS, PUSH, VOID, WIN, Leg, grade

SETTLEMENTS = ("reduce", "void_parlay", "push_loses")


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def grade_all(legs: list[Leg], game) -> np.ndarray:
    return np.column_stack([grade(l, game) for l in legs]) if legs else np.zeros((game.n_sims, 0))


def joint_from_grades(G: np.ndarray) -> dict:
    n = G.shape[0]
    k = int((G == WIN).all(1).sum())
    lo, hi = wilson_ci(k, n)
    return {"K": k, "N": n, "p": k / n, "ci95": (lo, hi),
            "mc_se": float(np.sqrt(max(k / n * (1 - k / n), 0) / n))}


def pairwise_corr(G: np.ndarray) -> np.ndarray:
    W = (G == WIN).astype(float)
    L = W.shape[1]
    C = np.full((L, L), np.nan)
    sd = W.std(0)
    for i in range(L):
        for j in range(L):
            if sd[i] > 0 and sd[j] > 0:
                C[i, j] = np.corrcoef(W[:, i], W[:, j])[0, 1]
    return C


def settle(G: np.ndarray, settlement: str = "reduce", leg_decimal: np.ndarray | None = None,
           parlay_decimal: float | None = None) -> dict:
    """Per-simulation gross return per unit stake. Returns NaN payouts where the
    reduced-odds payout cannot be computed (no per-leg odds given)."""
    if settlement not in SETTLEMENTS:
        raise ValueError(f"settlement must be one of {SETTLEMENTS}")
    n, L = G.shape
    lost = (G == LOSS).any(1)
    pv = (G == PUSH) | (G == VOID)
    any_pv = pv.any(1)
    all_pv = pv.all(1)
    full = (G == WIN).all(1)
    pay = np.zeros(n)
    if parlay_decimal is None and leg_decimal is not None:
        parlay_decimal = float(np.prod(leg_decimal))
    pay[full] = parlay_decimal if parlay_decimal is not None else np.nan
    reduced = ~lost & any_pv & ~all_pv
    refund = ~lost & all_pv
    if settlement == "reduce":
        pay[refund] = 1.0
        if leg_decimal is not None:
            dec = np.where(G == WIN, leg_decimal[None, :], 1.0)
            pay[reduced] = dec[reduced].prod(1)
        else:
            pay[reduced] = np.nan
    elif settlement == "void_parlay":
        pay[~lost & any_pv] = 1.0
        reduced = np.zeros(n, bool)
        refund = ~lost & any_pv
    else:  # push_loses
        pay[any_pv] = 0.0
        reduced = np.zeros(n, bool)
        refund = np.zeros(n, bool)
        lost = lost | any_pv
    return {"payout": pay, "p_full_win": float(full.mean()), "p_reduced_win": float(reduced.mean()),
            "p_refund": float(refund.mean()), "p_loss": float(lost.mean())}


@dataclass
class ParlayResult:
    legs: list
    leg_rates: list            # dicts per leg
    joint: dict                # K, N, p, ci95, mc_se
    independence_benchmark: float
    corr: np.ndarray
    settlement: dict
    fair: dict | None
    offered: dict | None
    ev: dict | None
    model_uncertainty: dict | None
    missing: list
    notes: list

    def summary(self) -> str:
        lines = []
        for l, r in zip(self.legs, self.leg_rates):
            lines.append(f"  {l.name():45s} win {r['p_win']:.4f}  push {r['p_push']:.4f}  "
                         f"void {r['p_void']:.4f}")
        j = self.joint
        lines.append(f"Joint P(all legs win) = K/N = {j['K']}/{j['N']} = {j['p']:.4f}  "
                     f"95% Monte Carlo CI [{j['ci95'][0]:.4f}, {j['ci95'][1]:.4f}]")
        lines.append(f"Independence benchmark (product of leg win rates, comparison only): "
                     f"{self.independence_benchmark:.4f}")
        if self.fair:
            lines.append(f"Fair odds: {self.fair['american']:+.0f} (decimal {self.fair['decimal']:.2f})")
        if self.offered:
            lines.append(f"Offered: {self.offered['american']:+.0f} (decimal {self.offered['decimal']:.2f}, "
                         f"implied {self.offered['implied']:.4f})")
        if self.ev:
            lines.append(f"Expected profit per unit ({self.settlement['rule']}): "
                         f"{self.ev['ev_per_unit']:+.4f}  (Monte Carlo SE {self.ev['mc_se']:.4f})")
        mu = self.model_uncertainty
        if mu:
            lines.append(f"Model uncertainty ({mu['B']} bootstrap refits x {mu['n_sims']} sims): "
                         f"joint p mean {mu['mean']:.4f}, sd {mu['sd']:.4f} "
                         f"(sd after removing Monte Carlo noise {mu['model_sd']:.4f}), "
                         f"5-95% [{mu['q05']:.4f}, {mu['q95']:.4f}]")
            lines.append("  More simulations shrink the Monte Carlo CI but not the model uncertainty.")
        for n_ in self.notes:
            lines.append("Note: " + n_)
        return "\n".join(lines)


def evaluate(game, legs: list[Leg], settlement: str = "reduce",
             leg_odds: list[float] | None = None, parlay_odds: float | None = None,
             model_uncertainty: dict | None = None) -> ParlayResult:
    """Evaluate a parlay on a SimulatedGame. Odds are American."""
    G = grade_all(legs, game)
    n = G.shape[0]
    rates = []
    for i in range(len(legs)):
        g = G[:, i]
        w, p, v = (g == WIN).mean(), (g == PUSH).mean(), (g == VOID).mean()
        decided = 1 - p - v
        rates.append({"p_win": float(w), "p_loss": float((g == LOSS).mean()), "p_push": float(p),
                      "p_void": float(v),
                      "p_win_given_decided": float(w / decided) if decided > 0 else np.nan,
                      "ci95": wilson_ci(int((g == WIN).sum()), n)})
    joint = joint_from_grades(G)
    indep = float(np.prod([r["p_win"] for r in rates]))
    notes = []
    leg_dec = np.asarray(O.american_to_decimal(leg_odds), float) if leg_odds is not None else None
    par_dec = float(O.american_to_decimal(parlay_odds)) if parlay_odds is not None else None
    st = settle(G, settlement, leg_dec, par_dec)
    st["rule"] = settlement
    fair = O.fair_odds(joint["p"]) if 0 < joint["p"] < 1 else None
    if joint["K"] == 0:
        notes.append("No simulation had every leg win; increase N or the legs are near-impossible together.")
    offered, ev = None, None
    if par_dec is not None or leg_dec is not None:
        d = par_dec if par_dec is not None else float(np.prod(leg_dec))
        offered = {"decimal": d, "american": float(O.decimal_to_american(d)), "implied": 1 / d}
        if np.isnan(st["payout"]).any():
            notes.append("Some simulations have pushed/void legs under 'reduce' settlement; "
                         "per-leg offered odds are needed to price them, so EV is not computed.")
        else:
            ev = O.expected_value_payouts(st["payout"])
    if any(r["p_void"] == 1 for r in rates):
        notes.append("At least one leg is void in every simulation (player not active in this prediction).")
    payout = st.pop("payout")
    st["payout_mean"] = float(np.nanmean(payout)) if not np.isnan(payout).all() else np.nan
    return ParlayResult(legs, rates, joint, indep, pairwise_corr(G), st, fair, offered, ev,
                        model_uncertainty, list(game.missing), notes)


def bootstrap_model_uncertainty(ctx, game_id: str, legs: list[Leg], B: int = 20,
                                n_sims: int = 2000, seed: int = 0, cutoff=None,
                                mode: str = "independent", home_spread: float | None = None,
                                total: float | None = None, qb_exits: bool = True) -> dict:
    """Refit the drive model and player usage models on bootstrap resamples, rerun
    the simulation for each, and report the spread of the joint probability.
    In market_anchored mode every refit is re-anchored to the same line, so the
    spread reflects model uncertainty in the joint structure given the market."""
    from .pipeline import simulate_game
    rng = np.random.default_rng(seed)
    ps, legp = [], []
    for b in range(B):
        g = simulate_game(ctx, game_id, n_sims, seed=int(rng.integers(1 << 31)), cutoff=cutoff,
                          bootstrap_rng=np.random.default_rng(int(rng.integers(1 << 31))),
                          mode=mode, home_spread=home_spread, total=total, qb_exits=qb_exits)
        G = grade_all(legs, g)
        ps.append((G == WIN).all(1).mean())
        legp.append((G == WIN).mean(0))
    ps = np.array(ps)
    mc_var = float(np.mean(ps * (1 - ps) / n_sims))
    tot_var = float(ps.var(ddof=1)) if B > 1 else np.nan
    return {"B": B, "n_sims": n_sims, "values": ps.tolist(), "mean": float(ps.mean()),
            "sd": float(np.sqrt(tot_var)), "mc_sd_within": float(np.sqrt(mc_var)),
            "model_sd": float(np.sqrt(max(tot_var - mc_var, 0.0))),
            "q05": float(np.quantile(ps, 0.05)), "q95": float(np.quantile(ps, 0.95)),
            "leg_sd": np.array(legp).std(0, ddof=1).tolist()}


def parse_leg(spec: str, game, names: dict) -> Leg:
    """Leg strings:
      <market>:<player>:<over|under>:<line>   e.g. receiving_yards:Travis Kelce:over:55.5
      anytime_td:<player>:<yes|no>
      spread:<TEAM>:<line>  |  moneyline:<TEAM>  |  total:<over|under>:<line>
    <player> is a gsis_id or a (unique) name among the two teams' pregame rosters."""
    parts = spec.split(":")
    m = parts[0]
    if m == "spread":
        return Leg("spread", parts[1], float(parts[2]))
    if m == "moneyline":
        return Leg("moneyline", parts[1])
    if m == "total":
        return Leg("total", parts[1], float(parts[2]))
    who = parts[1]
    roster = {pid for tp in game.players.teams.values() for pid in tp.ids}
    if who not in roster:
        hits = [pid for pid in roster if who.lower() in names.get(pid, "").lower()]
        if len(hits) != 1:
            # not on either pregame roster (e.g. ruled out) -> keep the id so the leg is void
            hits = [pid for pid, nm in names.items() if who.lower() == nm.lower()][:1] or [who]
        who = hits[0]
    label = f"{names.get(who, who)} {m} {parts[2]}" + (f" {parts[3]}" if len(parts) > 3 else "")
    if m == "anytime_td":
        return Leg(m, parts[2], None, who, label)
    return Leg(m, parts[2], float(parts[3]), who, label)


def main(argv=None):
    import argparse

    from .pipeline import load_context, player_names, simulate_game
    ap = argparse.ArgumentParser(description="Price a same-game parlay from shared simulations")
    ap.add_argument("--game-id", required=True)
    ap.add_argument("--leg", action="append", required=True, help="see parse_leg")
    ap.add_argument("--leg-odds", type=float, nargs="*", help="offered American odds per leg")
    ap.add_argument("--parlay-odds", type=float, help="offered American odds for the parlay")
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--settlement", default="reduce", choices=SETTLEMENTS)
    ap.add_argument("--bootstrap", type=int, default=20, help="model refits (0 = skip)")
    ap.add_argument("--bootstrap-sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="independent", choices=["independent", "market_anchored"])
    ap.add_argument("--spread", type=float, help="home team's sportsbook spread for market_anchored "
                    "(negative = home favored, e.g. -3.5); default: nflverse line")
    ap.add_argument("--total", type=float, help="game total for market_anchored; default: nflverse line")
    ap.add_argument("--qb-exits", default="on", choices=["on", "off"],
                    help="simulate starting-QB early exits (default on)")
    a = ap.parse_args(argv)
    qx = a.qb_exits == "on"
    ctx = load_context()
    names = player_names()
    game = simulate_game(ctx, a.game_id, a.sims, seed=a.seed, mode=a.mode,
                         home_spread=a.spread, total=a.total, qb_exits=qx)
    legs = [parse_leg(s, game, names) for s in a.leg]
    mu = bootstrap_model_uncertainty(ctx, a.game_id, legs, a.bootstrap, a.bootstrap_sims,
                                     seed=a.seed + 1, mode=game.anchor["mode"],
                                     home_spread=a.spread, total=a.total, qb_exits=qx) if a.bootstrap else None
    res = evaluate(game, legs, a.settlement, a.leg_odds, a.parlay_odds, mu)
    print(f"{game.game_id} ({game.away_team} @ {game.home_team}, {game.gameday:%Y-%m-%d}); "
          f"models fit on data before {game.team_model.cutoff:%Y-%m-%d}; N = {a.sims}")
    print(res.summary())
    if len(legs) > 1:
        print("Pairwise correlation of leg wins (from the shared simulations):")
        for i, l in enumerate(legs):
            row = "  ".join(f"{c:+.3f}" if np.isfinite(c) else "  nan " for c in res.corr[i])
            print(f"  {i + 1}. {l.name()[:40]:40s} {row}")
    print("Missing data / assumptions used in this prediction:")
    for m in res.missing:
        print("  - " + m)


if __name__ == "__main__":
    main()
