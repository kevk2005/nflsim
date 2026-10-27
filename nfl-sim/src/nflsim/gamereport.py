"""Game report (Milestone 4): everything the simulator says happens in one game,
with the data behind each number. Saved odds only, never pulls.

Per game, exported to reports/<game_id>/ (CSV tables + report.html + meta.json):
  team_dist.csv          spread / total / points / drives / plays / pass rate percentiles
  team_probs.csv         win and 20+/27+/34+ scoring chances (MC 95% + model range)
  player_stats.csv       per player and stat: mean, 10th/25th/50th/75th/90th percentiles
  player_thresholds.csv  hit % at every grid threshold: MC 95%, model range, role
                         baseline, own (shrunk) rate, lift, market % (info), trust label
  player_history.csv     each player's last `window_games` games played
  player_usage.csv       the model inputs behind his numbers, vs the role/position average
  market_lines.csv       saved prop lines (info only) with the sim % at the same line
  correlations.csv       correlations between legs across simulations

Model range = p_sim +/- 1.96 x model sd, where model sd comes from `B` bootstrap
refits (drive model + every player's usage history resampled) with the Monte
Carlo part of their spread removed.

CLI: python -m nflsim.gamereport build [--days 7] [--game-id ...] [--sims 10000] [--bootstrap 10]
"""
from __future__ import annotations

import argparse
import html
import json
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import tails as T
from .ingest import PROCESSED

REPORTS = Path(__file__).resolve().parents[2] / "reports"
TABLES = ("team_dist", "team_probs", "player_stats", "player_thresholds", "player_history", "player_usage",
          "market_lines", "correlations")
SCORE_LINES = (20, 27, 34)
Z = 1.959964


def _pcts(x) -> dict:
    x = np.asarray(x, float)
    q = np.percentile(x, T.PCTS)
    return {"mean": float(x.mean()), **{f"p{p}": float(v) for p, v in zip(T.PCTS, q)}}


def _team_series(sg) -> dict:
    """[n_sims] arrays for the team / game section."""
    s, h, a = sg.sim, sg.home_team, sg.away_team
    tp = sg.players.team_plays
    out = {f"{h} margin (home − away)": s.margin, "total points": s.total,
           f"{h} points": s.home_score, f"{a} points": s.away_score,
           f"{h} drives": s.home_drives, f"{a} drives": s.n_drives - s.home_drives}
    for side, t in ((0, h), (1, a)):
        out[f"{t} plays"] = tp[side]["plays"]
        db, ru = tp[side]["dropbacks"], tp[side]["designed_runs"]
        out[f"{t} pass rate"] = db / np.maximum(db + ru, 1)
    return out


def _team_probs(sg) -> dict:
    s, h, a = sg.sim, sg.home_team, sg.away_team
    out = {f"{h} win": s.margin > 0, f"{a} win": s.margin < 0, "tie": s.margin == 0}
    for t, pts in ((h, s.home_score), (a, s.away_score)):
        for line in SCORE_LINES:
            out[f"{t} scores {line}+"] = pts >= line
    return {k: float(np.mean(v)) for k, v in out.items()}


def _model_sd(boot: pd.DataFrame, boot_sims: int) -> pd.Series:
    """Bootstrap sd with the Monte Carlo variance of each refit removed."""
    m = boot.mean(axis=1)
    var = boot.var(axis=1, ddof=1) - m * (1 - m) / boot_sims
    return np.sqrt(np.maximum(var, 0))


def _ratio_text(name, v, ref, fmt="{:.0%}", ref_name=""):
    r = v / ref if ref and ref > 0 else np.nan
    return {"input": name, "value": v, "reference": ref, "ratio": r,
            "text": f"{name} {fmt.format(v)} vs {fmt.format(ref)}{ref_name}" + ("" if not np.isfinite(r) else f" (×{r:.2f})")}


def usage_rows(sg, k, side, tp, pri, role_key, fr) -> list:
    """The model's actual inputs for player k, each next to the average it is shrunk toward."""
    from .players import _role_key
    pos = tp.position[k]
    rk = _role_key(pos, tp.rank[k])
    rname = T.role_name(pos, tp.rank[k])
    pp = pri.get(pos)
    st = sg.players.stats[side]
    act = sg.players.active[side][:, k]
    tpl = sg.players.team_plays[side]
    rows = []
    if pos == "QB" and k == tp.qb_idx:
        pref = "home" if side == 0 else "away"
        opp = "away" if side == 0 else "home"
        rows += [
            {"input": "passing volume and efficiency", "value": np.nan, "reference": np.nan, "ratio": np.nan,
             "text": "He throws every simulated pass of his team's drives. Completion rate and yards per attempt "
                     "come from historical drives with the same outcome and situation, not from his own rates; "
                     "his quality enters through the team drive model (QB EPA below)."},
            {"input": "QB EPA/play (drive model)", "value": fr.get(f"{pref}_qb_epa"), "reference": np.nan,
             "ratio": np.nan, "text": f"QB EPA/play {fr.get(f'{pref}_qb_epa', np.nan):+.3f} "
                                      f"(change vs team baseline {fr.get(f'{pref}_qb_delta', np.nan):+.3f})"},
            {"input": "team offense EPA/play (drive model)", "value": fr.get(f"{pref}_off_epa"),
             "reference": fr.get("lg_epa"), "ratio": np.nan,
             "text": f"team offense EPA/play {fr.get(f'{pref}_off_epa', np.nan):+.3f} "
                     f"vs league {fr.get('lg_epa', np.nan):+.3f}; opponent defense allows "
                     f"{fr.get(f'{opp}_def_epa', np.nan):+.3f}"},
            {"input": "team dropbacks (sim mean)", "value": float(tpl["dropbacks"].mean()), "reference": np.nan,
             "ratio": np.nan, "text": f"team dropbacks per game {tpl['dropbacks'].mean():.1f} (sim mean)"},
        ]
    if tp.tgt_share[k] > 0:
        rows.append(_ratio_text("target share", tp.tgt_share[k], pri.tshare.get(rk, np.nan), ref_name=f" for {rname}"))
        rows.append(_ratio_text("catch rate", tp.catch[k], pp["catch"], ref_name=f" for {pos}s"))
        rows.append(_ratio_text("catches of 20+ yds", tp.rec_bins[k][5:].sum(), pp["rec_bins"][5:].sum(),
                                ref_name=f" for {pos}s"))
        rows.append({"input": "receiving TD rate multiplier", "value": tp.rec_td_ratio[k], "reference": 1.0,
                     "ratio": tp.rec_td_ratio[k],
                     "text": f"receiving TD rate ×{tp.rec_td_ratio[k]:.2f} of the {pos} average "
                             "(stands in for red-zone usage; the model has no red-zone share)"})
    if tp.car_share[k] > 0:
        rows.append(_ratio_text("carry share", tp.car_share[k], pri.cshare.get(rk, np.nan), ref_name=f" for {rname}"))
        rows.append(_ratio_text("carries of 10+ yds", tp.rush_bins[k][4:].sum(), pp["rush_bins"][4:].sum(),
                                ref_name=f" for {pos}s"))
        rows.append({"input": "rushing TD rate multiplier", "value": tp.rush_td_ratio[k], "reference": 1.0,
                     "ratio": tp.rush_td_ratio[k],
                     "text": f"rushing TD rate ×{tp.rush_td_ratio[k]:.2f} of the {pos} average"})
    pa = float(tp.p_active[k]) if tp.p_active is not None else 1.0
    rows.append({"input": "P(active)", "value": pa, "reference": float(pri.active.get(rk, np.nan)), "ratio": np.nan,
                 "text": f"P(active) {pa:.0%} (role {rname}: {pri.active.get(rk, np.nan):.0%})"})
    if tp.tgt_share[k] > 0:
        rows.append({"input": "targets (sim mean)", "value": float(st["targets"][act, k].mean()), "reference": np.nan,
                     "ratio": np.nan, "text": f"{st['targets'][act, k].mean():.1f} targets of the team's "
                                              f"{tpl['targets'].mean():.1f} (sim means)"})
    if tp.car_share[k] > 0:
        rows.append({"input": "carries (sim mean)", "value": float(st["rush_att"][act, k].mean()),
                     "reference": np.nan, "ratio": np.nan,
                     "text": f"{st['rush_att'][act, k].mean():.1f} carries of the team's "
                             f"{tpl['designed_runs'].mean():.1f} designed runs (sim means)"})
    return rows


def drivers_text(rows: list) -> str:
    """The inputs furthest from their averages (largest |log ratio|), in words.
    Shares under 3% on both sides (e.g. a receiver's carry share) are ignored."""
    r = [x for x in rows if np.isfinite(x.get("ratio", np.nan)) and x["ratio"] > 0
         and not (x["input"].endswith("share") and max(x["value"], x.get("reference", 0) or 0) < 0.03)]
    r = sorted(r, key=lambda x: -abs(np.log(x["ratio"])))[:2]
    r = [x for x in r if abs(np.log(x["ratio"])) > np.log(1.1)]
    if not r:
        return "close to the role/position average on every input; volume comes from the team's simulated plays"
    return "; ".join(x["text"] for x in r)


def correlations(sg, stats_rows: pd.DataFrame, names: dict, top: int) -> pd.DataFrame:
    """Pearson correlations across simulations between game legs and main players'
    stats (a player's value is 0 in simulations where he sits)."""
    s, h, a = sg.sim, sg.home_team, sg.away_team
    legs = {f"{h} win": ("", h, (s.margin > 0).astype(float)), f"{a} win": ("", a, (s.margin < 0).astype(float)),
            "total points": ("", "", s.total.astype(float)),
            f"{h} points": ("", h, s.home_score.astype(float)), f"{a} points": ("", a, s.away_score.astype(float))}
    want = {"QB": [("pass_yards", 0), ("pass_td", 0), ("rush_yards", 10)],
            "RB": [("rush_att", 3), ("rush_yards", 15), ("receptions", 1.5), ("rec_yards", 10), ("tds", 0.2)],
            "WR": [("receptions", 2), ("rec_yards", 15), ("tds", 0.2)],
            "TE": [("receptions", 2), ("rec_yards", 15), ("tds", 0.2)]}
    main = stats_rows[stats_rows["p_active"] >= 0.5]
    for (pid, team, pos), x in main.groupby(["player_id", "team", "position"]):
        means = dict(zip(x["stat"], x["mean"]))
        side = 0 if team == h else 1
        tp = sg.players.teams[side]
        k = tp.ids.index(pid)
        st = sg.players.stats[side]
        for stat, min_mean in want.get(pos, []):
            if stat in means and means[stat] >= min_mean:
                legs[f"{names.get(pid, pid)} {T.STAT_LABELS[stat]}"] = (pid, team, T.stat_values(st, k, stat).astype(float))
    keys = [k_ for k_, v in legs.items() if v[2].std() > 0]
    M = np.corrcoef(np.vstack([legs[k_][2] for k_ in keys]))
    rows = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            (pi, ti, _), (pj, tj, _) = legs[keys[i]], legs[keys[j]]
            if pi and pj and pi == pj:
                kind = "same player"
            elif not pi and not pj:
                kind = "game level"
            elif (keys[i].endswith(" win") and not pi and pj and tj != ti) or \
                    (keys[j].endswith(" win") and not pj and pi and ti != tj):
                kind = "opponent win (mirror)"
            else:
                kind = "cross"
            rows.append({"leg_a": keys[i], "leg_b": keys[j], "corr": float(M[i, j]), "kind": kind})
    d = pd.DataFrame(rows).sort_values("corr", ascending=False, key=abs).reset_index(drop=True)
    cross = d[d["kind"] == "cross"]
    d["top"] = ""
    d.loc[cross.nlargest(top, "corr").index, "top"] = "positive"
    d.loc[cross.nsmallest(top, "corr").query("corr < 0").index, "top"] = "negative"
    return d


def build(ctx, base, cfg, game_id: str, n_sims: int = 10000, B: int = 10, boot_sims: int = 2000, seed: int = 0,
          now=None, names: dict | None = None, rep: dict | None = None, verbose: bool = True) -> dict:
    from . import odds_feed as F
    from .pipeline import simulate_game
    best = T.best_config()
    mode, qbx = best["mode"], best["qb_exits"] == "on"
    names = names or {}
    row = ctx.gf[ctx.gf["game_id"] == game_id].iloc[0]
    cutoff, season = pd.Timestamp(row["gameday"]), int(row["season"])
    now = pd.Timestamp(now) if now is not None else F.utcnow()
    rng = np.random.default_rng([seed, zlib.crc32(game_id.encode())])
    t0 = time.time()
    sg = simulate_game(ctx, game_id, n_sims, seed=int(rng.integers(1 << 31)), mode=mode, qb_exits=qbx,
                       pass_shock=0.0, as_of=now)
    h, a = sg.home_team, sg.away_team
    mlook = T._market_lookup(mt := T.market_table([game_id], as_of=now))
    tr, sr = T.game_rows(sg, cfg, base, cutoff, season, mlook=mlook)
    thr, sts = pd.DataFrame(tr), pd.DataFrame(sr)
    probs = _team_probs(sg)
    # bootstrap model ranges
    # bootstrap model ranges (same player-inclusion rule, scaled to the smaller runs)
    bcfg = {**cfg, "report": {**cfg["report"], "min_active_sims": max(
        10, round(cfg["report"]["min_active_sims"] * boot_sims / n_sims))}}
    key = pd.MultiIndex.from_frame(thr[["player_id", "stat", "at_least"]])
    boots_p, boots_t = [], []
    for b in range(B):
        sb = simulate_game(ctx, game_id, boot_sims, seed=int(rng.integers(1 << 31)), mode=mode, qb_exits=qbx,
                           pass_shock=0.0, as_of=now, bootstrap_rng=np.random.default_rng(int(rng.integers(1 << 31))))
        btr, _ = T.game_rows(sb, bcfg, base, cutoff, season)
        bp = pd.DataFrame(btr).set_index(["player_id", "stat", "at_least"])["p_sim"].reindex(key)
        # a tracked player's stat that is zero in every simulation was left out: its hit % is 0
        tracked = {tp.ids[k] for _, k, tp, _ in T.tracked_players(sb, bcfg["report"]["min_active_sims"])}
        bp[bp.isna() & key.get_level_values(0).isin(tracked)] = 0.0
        boots_p.append(bp)
        boots_t.append(pd.Series(_team_probs(sb)))
    if B >= 2:
        thr["model_sd"] = _model_sd(pd.concat(boots_p, axis=1), boot_sims).to_numpy()
        tsd = _model_sd(pd.concat(boots_t, axis=1), boot_sims)
    else:
        thr["model_sd"] = np.nan
        tsd = pd.Series(np.nan, index=list(probs))
    thr["model_lo"] = (thr["p_sim"] - Z * thr["model_sd"]).clip(0, 1)
    thr["model_hi"] = (thr["p_sim"] + Z * thr["model_sd"]).clip(0, 1)
    # team section
    tdist = pd.DataFrame([{"metric": k, **_pcts(v)} for k, v in _team_series(sg).items()])
    tprobs = []
    for k, p in probs.items():
        lo, hi = T.wilson(round(p * n_sims), n_sims)
        sd = float(tsd.get(k, np.nan))
        tprobs.append({"outcome": k, "p": p, "mc_lo": float(lo), "mc_hi": float(hi), "model_sd": sd,
                       "model_lo": max(0.0, p - Z * sd) if np.isfinite(sd) else np.nan,
                       "model_hi": min(1.0, p + Z * sd) if np.isfinite(sd) else np.nan})
    tprobs = pd.DataFrame(tprobs)
    # names, trust labels
    label = T.trust_labels(rep)
    for d in (thr, sts):
        d.insert(3, "name", [names.get(p, p) for p in d["player_id"]])
    thr["trust"] = [label(s, lv) for s, lv in zip(thr["stat"], thr["level"])]
    # data behind each number: history, usage, market lines
    pri = ctx.priors(cutoff, season)
    gm = ctx.games.set_index("game_id")
    hist_rows, use_rows, drivers = [], [], {}
    fr = row.to_dict()
    for side, k, tp, usage in T.tracked_players(sg, cfg["report"]["min_active_sims"]):
        pid = tp.ids[k]
        hx = base.history(pid, cutoff)
        for r in hx.itertuples():
            g = gm.loc[r.game_id] if r.game_id in gm.index else None
            opp = "" if g is None else (g["away_team"] if g["home_team"] == r.team else g["home_team"])
            hist_rows.append({"player_id": pid, "name": names.get(pid, pid), "team": tp.team,
                              "gameday": pd.Timestamp(r.gameday).date(), "game_id": r.game_id, "for": r.team,
                              "opponent": opp, "role_then": r.role, **{s: getattr(r, s) for s in T.STAT_COLS}})
        ur = usage_rows(sg, k, side, tp, pri, None, fr)
        drivers[pid] = drivers_text(ur)
        use_rows += [{"player_id": pid, "name": names.get(pid, pid), "team": tp.team, **u} for u in ur]
    sts["drivers"] = sts["player_id"].map(drivers)
    sts["n_hist_games"] = sts["player_id"].map(thr.drop_duplicates("player_id").set_index("player_id")["n_hist_games"])
    ml = []
    for r in mt.itertuples():
        stat = T.MARKET_STAT.get(r.market)
        x = None
        for side, tp in sg.players.teams.items():
            if r.player_id in tp.ids and stat:
                k = tp.ids.index(r.player_id)
                x = T.stat_values(sg.players.stats[side], k, stat)[sg.players.active[side][:, k]]
        line = 0.5 if r.market == "anytime_td" else r.line
        ml.append({"player_id": r.player_id, "name": names.get(r.player_id, r.player_id), "market": r.market,
                   "stat": stat, "line": r.line, "p_market": r.p_market, "one_sided": r.one_sided,
                   "n_books": r.n_books, "best_price": r.best_price,
                   "p_sim_over": None if x is None or not len(x) else float((x > line).mean())})
    ml = pd.DataFrame(ml, columns=["player_id", "name", "market", "stat", "line", "p_market", "one_sided",
                                   "n_books", "best_price", "p_sim_over"])
    corr = correlations(sg, sts, names, cfg["report"]["corr_top"])
    opp_ctx = []
    for side, (t, o) in enumerate(((h, a), (a, h))):
        me, op = ("home", "away") if side == 0 else ("away", "home")
        opp_ctx.append({"team": t, "opponent": o, "off_epa": fr.get(f"{me}_off_epa"), "off_sr": fr.get(f"{me}_off_sr"),
                        "qb_epa": fr.get(f"{me}_qb_epa"), "opp_def_epa_allowed": fr.get(f"{op}_def_epa"),
                        "opp_def_sr_allowed": fr.get(f"{op}_def_sr"), "lg_epa": fr.get("lg_epa"),
                        "lg_sr": fr.get("lg_sr")})
    sched = F.load_schedule()
    ko = sched.loc[sched["game_id"] == game_id, "kickoff_utc"]
    meta = {"game_id": game_id, "home": h, "away": a, "matchup": f"{a} @ {h}",
            "kickoff_utc": None if ko.empty else str(ko.iloc[0]), "built_at": str(now),
            "n_sims": n_sims, "bootstrap": B, "boot_sims": boot_sims, "mode": mode, "qb_exits": qbx,
            "pass_shock": "off", "anchor": {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
                                            for k, v in (sg.anchor or {}).items()},
            "history_window_games": base.window, "role_prior_seasons": base.prior_seasons,
            "shrink_prior_games": base.k, "notes": sg.missing,
            "opponent_context": opp_ctx,
            "opponent_note": ("The player model does not use the opponent's defense against his position "
                              "(not modeled, not added). The opponent's overall defense (opponent-adjusted EPA "
                              "and success rate allowed per play) enters the team drive model, which sets his "
                              "team's drives, plays and scoring, and so his volume and TD chances."),
            "team_range_note": ("Market-anchored: every refit is re-anchored to the same market margin and "
                                "total, so the team-level model ranges are near zero by construction; they do not "
                                "mean the model is certain." if (sg.anchor or {}).get("mode") == "market_anchored"
                                else ""),
            "keep_test": None if rep is None else rep.get("keep_test"),
            "seconds": round(time.time() - t0, 1)}
    if verbose:
        print(f"{game_id}: {len(sts['player_id'].unique())} players, {len(thr)} thresholds, "
              f"{meta['seconds']}s", flush=True)
    return {"meta": meta, "team_dist": tdist, "team_probs": tprobs, "player_stats": sts,
            "player_thresholds": thr, "player_history": pd.DataFrame(hist_rows), "player_usage": pd.DataFrame(use_rows),
            "market_lines": ml, "correlations": corr}


# --------------------------------------------------------------------------- #
# Export / load
# --------------------------------------------------------------------------- #
def export(rep: dict, root: Path | None = None) -> Path:
    out = (root or REPORTS) / rep["meta"]["game_id"]
    out.mkdir(parents=True, exist_ok=True)
    for t in TABLES:
        rep[t].to_csv(out / f"{t}.csv", index=False)
    (out / "meta.json").write_text(json.dumps(rep["meta"], indent=1, default=str))
    (out / "report.html").write_text(to_html(rep))
    return out


def load(game_id: str, root: Path | None = None) -> dict | None:
    d = (root or REPORTS) / game_id
    if not (d / "meta.json").exists():
        return None
    rep = {"meta": json.loads((d / "meta.json").read_text())}
    for t in TABLES:
        p = d / f"{t}.csv"
        rep[t] = pd.read_csv(p) if p.exists() and p.stat().st_size > 1 else pd.DataFrame()
    if "top" in rep["correlations"]:
        rep["correlations"]["top"] = rep["correlations"]["top"].fillna("")
    return rep


def saved_reports(root: Path | None = None) -> list:
    r = root or REPORTS
    return sorted(p.name for p in r.iterdir() if (p / "meta.json").exists()) if r.exists() else []


def _fmt_pct(p):
    return "" if p is None or pd.isna(p) else f"{100 * p:.1f}%"


def _table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, escape=True, border=0, classes="t", na_rep="")


def headline_thresholds(thr: pd.DataFrame, pid: str, n: int = 6) -> pd.DataFrame:
    """A player's thresholds nearest 50/25/10% per stat: a compact view of his range."""
    x = thr[thr["player_id"] == pid]
    out = []
    for stat, g in x.groupby("stat", sort=False):
        for target in (0.5, 0.25, 0.10):
            out.append(g.iloc[(g["p_sim"] - target).abs().argmin()])
    return pd.DataFrame(out).drop_duplicates(["stat", "at_least"])


def to_html(rep: dict) -> str:
    m = rep["meta"]
    tp = rep["team_probs"].copy()
    td = rep["team_dist"].copy()
    sts, thr = rep["player_stats"], rep["player_thresholds"]
    css = """body{font:14px/1.45 -apple-system,system-ui,sans-serif;margin:24px auto;max-width:1100px;padding:0 16px;
color:#1f1f1e;background:#fbfbfa}h1{font-size:22px}h2{font-size:18px;margin-top:28px;border-bottom:1px solid #ddd}
h3{font-size:15px;margin:18px 0 4px}.t{border-collapse:collapse;margin:6px 0 10px;font-variant-numeric:tabular-nums}
.t td,.t th{padding:3px 8px;border-bottom:1px solid #eee;text-align:right}.t th{background:#f1f0ee}
.t td:first-child,.t th:first-child{text-align:left}.note{color:#5f5e5b;font-size:13px}.warn{color:#9a3b12}
@media (prefers-color-scheme:dark){body{background:#1b1b1a;color:#e8e6e1}.t th{background:#2a2a28}
.t td{border-color:#333}.note{color:#a3a19b}}"""
    a = m.get("anchor") or {}
    anchor = (f"market-anchored to home margin {a.get('target_margin')} and total {a.get('target_total')} "
              f"({a.get('source')})" if a.get("mode") == "market_anchored" else
              f"independent ({a.get('fallback', 'no market anchor')})")
    kt = m.get("keep_test")
    parts = [f"<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
             f"<title>{html.escape(m['matchup'])} report</title><style>{css}</style></head><body>",
             f"<h1>{html.escape(m['matchup'])} — game report</h1>",
             f"<p class='note'>Kickoff {m['kickoff_utc']} · built {m['built_at'][:16]} · {m['n_sims']:,} simulations · "
             f"{m['bootstrap']} bootstrap refits × {m['boot_sims']:,} · mode: {html.escape(anchor)} · QB exits "
             f"{'on' if m['qb_exits'] else 'off'} · pass shock off. Saved odds only.</p>"]
    if kt:
        parts.append(f"<p class='note'>Backtest keep test (2024 → 2026 wk 7): <b>{kt['overall']}</b>. Calibration "
                     f"by level {', '.join(f'{k} {v}' for k, v in kt['a_calibration_by_level'].items())}; percentile check {kt['b_percentiles_pooled']}. "
                     "Rows marked “not trusted” below failed it.</p>")
    else:
        parts.append("<p class='warn'>No backtest report: nothing here has been validated.</p>")
    for c in ("p", "mc_lo", "mc_hi", "model_lo", "model_hi"):
        tp[c] = tp[c].map(_fmt_pct)
    tp["MC 95%"] = tp["mc_lo"] + "–" + tp["mc_hi"]
    tp["model range"] = tp["model_lo"] + "–" + tp["model_hi"]
    parts += ["<h2>Game</h2>", _table(tp[["outcome", "p", "MC 95%", "model range"]].rename(columns={"p": "sim"})),
              f"<p class='note'>{html.escape(m.get('team_range_note', ''))}</p>",
              _table(td.round(2).rename(columns={"p10": "10th", "p25": "25th", "p50": "median", "p75": "75th",
                                                 "p90": "90th"}))]
    oc = pd.DataFrame(m["opponent_context"])
    parts += ["<h3>Team inputs and opponent context</h3>", _table(oc.round(3)),
              f"<p class='note'>{html.escape(m['opponent_note'])}</p>"]
    corr = rep["correlations"]
    if len(corr):
        top = corr[corr["top"] != ""].sort_values("corr", ascending=False)
        parts += ["<h2>What moves together</h2>",
                  "<p class='note'>Correlation across simulations (a player's value is 0 when he sits). "
                  "Same-player pairs and game-level identities are left out here; all pairs are in correlations.csv.</p>",
                  _table(top[["leg_a", "leg_b", "corr"]].round(2))]
    ml = rep["market_lines"]
    parts.append("<h2>Players</h2>")
    parts.append(f"<p class='note'>Percentiles over simulations where the player is active. Hit % thresholds shown "
                 f"are each stat's lines nearest 50/25/10%; every threshold is in player_thresholds.csv. Role "
                 f"baseline = players in the same pregame role, previous {m['role_prior_seasons']} seasons. Own rate = "
                 f"his last {m['history_window_games']} games played, shrunk toward the role baseline with "
                 f"{m['shrink_prior_games']:g} games of weight. Lift = sim − role. Market % is info only.</p>")
    for team in (m["away"], m["home"]):
        parts.append(f"<h2>{html.escape(team)}</h2>")
        st_t = sts[sts["team"] == team]
        order = st_t.groupby("player_id")["p_active"].first().sort_values(ascending=False).index
        for pid in order:
            x = st_t[st_t["player_id"] == pid]
            r0 = x.iloc[0]
            parts.append(f"<h3>{html.escape(str(r0['name']))} · {r0['position']} · role {r0['role']}"
                         f"{' · usage ' + str(r0['usage_role']) if isinstance(r0['usage_role'], str) and r0['usage_role'] else ''}"
                         f" · active {100 * r0['p_active']:.0f}%</h3>")
            parts.append(f"<p class='note'>Drivers: {html.escape(str(r0['drivers']))}. History: "
                         f"{int(r0['n_hist_games']) if pd.notna(r0['n_hist_games']) else 0} games.</p>")
            v = x[["stat", "mean", "p10", "p25", "p50", "p75", "p90"]].copy()
            v["stat"] = v["stat"].map(T.STAT_LABELS)
            parts.append(_table(v.round(1).rename(columns={"p10": "10th", "p25": "25th", "p50": "median",
                                                           "p75": "75th", "p90": "90th"})))
            hx = headline_thresholds(thr, pid)
            if len(hx):
                hv = pd.DataFrame({"outcome": [f"{T.STAT_LABELS[s]} ≥ {t}" for s, t in zip(hx["stat"], hx["at_least"])],
                                   "level": hx["level"], "sim": hx["p_sim"].map(_fmt_pct),
                                   "MC 95%": [f"{_fmt_pct(l_)}–{_fmt_pct(h_)}" for l_, h_ in zip(hx["mc_lo"], hx["mc_hi"])],
                                   "model range": [f"{_fmt_pct(l_)}–{_fmt_pct(h_)}" for l_, h_ in
                                                   zip(hx["model_lo"], hx["model_hi"])],
                                   "role": hx["p_role"].map(_fmt_pct), "own": hx["p_hist"].map(_fmt_pct),
                                   "lift": [f"{100 * v_:+.1f}" for v_ in hx["lift"]],
                                   "market": hx["p_market"].map(_fmt_pct) if "p_market" in hx else "",
                                   "trust": hx["trust"]})
                parts.append(_table(hv))
            mx = ml[ml["player_id"] == pid]
            if len(mx):
                mv = pd.DataFrame({"market (info)": mx["market"], "line": mx["line"],
                                   "market %": mx["p_market"].map(_fmt_pct), "sim %": mx["p_sim_over"].map(_fmt_pct),
                                   "books": mx["n_books"]})
                parts.append(_table(mv))
    parts.append("<h2>Not modeled / notes</h2><ul>" + "".join(f"<li>{html.escape(n)}</li>" for n in m["notes"])
                 + "</ul></body></html>")
    return "\n".join(parts)


def build_many(game_ids=None, days: int = 7, n_sims: int = 10000, B: int = 10, boot_sims: int = 2000,
               seed: int = 0, now=None, root: Path | None = None, verbose: bool = True) -> list:
    from . import odds_feed as F
    from .pipeline import player_names
    cfg = T.load_config()
    ctx, base = T.shared_inputs(cfg)
    now = pd.Timestamp(now) if now is not None else F.utcnow()
    if game_ids is None:
        sched = F.load_schedule()
        up = sched[(sched["kickoff_utc"] > now) & (sched["kickoff_utc"] <= now + pd.Timedelta(days=days))]
        game_ids = list(up["game_id"])
    names, rep, out = player_names(), T.load_report(), []
    for g in game_ids:
        if (ctx.gf["game_id"] == g).any():
            out.append(export(build(ctx, base, cfg, g, n_sims, B, boot_sims, seed, now, names, rep, verbose), root))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m nflsim.gamereport")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--days", type=int, default=7)
    b.add_argument("--game-id", nargs="*")
    b.add_argument("--sims", type=int, default=10000)
    b.add_argument("--bootstrap", type=int, default=10)
    b.add_argument("--boot-sims", type=int, default=2000)
    a = ap.parse_args(argv)
    paths = build_many(a.game_id, a.days, a.sims, a.bootstrap, a.boot_sims)
    print(f"{len(paths)} reports in {REPORTS}")


if __name__ == "__main__":
    main()
