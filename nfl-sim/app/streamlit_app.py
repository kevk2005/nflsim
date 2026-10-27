"""NFL Monte Carlo simulator: market browser, parlay builder, parlay log, backtest.

Run:  .venv/bin/streamlit run app/streamlit_app.py

Reads saved files only (odds snapshots, nflverse data, the parlay log, backtest
reports). It never calls The Odds API: new prices come only from the odds_feed
CLI, which has its own credit budget.
"""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from nflsim import app_support as A
from nflsim import odds as O
from nflsim.ingest import PROCESSED
from nflsim.markets import PUSH, VOID, WIN

st.set_page_config(page_title="NFL Sim Parlays", layout="wide")

MODEL_COLOR, INDEP_COLOR, REF_COLOR = "#2a78d6", "#eb6834", "#8a8984"   # categorical slots 1-2, neutral


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading data and models...")
def context():
    from nflsim.pipeline import load_context
    return load_context()


@st.cache_data(show_spinner=False, ttl=60)
def snapshots():
    return A.load_snapshots()      # refreshed every minute, or with "Reload snapshots"


@st.cache_data(show_spinner=False)
def names():
    from nflsim.pipeline import player_names
    return player_names()


@st.cache_resource(show_spinner="Simulating...", max_entries=8)
def simulated(game_id: str, n_sims: int, mode: str, qb_exits: bool, seed: int):
    from nflsim.pipeline import simulate_game
    return simulate_game(context(), game_id, n_sims, seed=seed, mode=mode, qb_exits=qb_exits,
                         pass_shock=0.0)


@st.cache_data(show_spinner="Refitting models on bootstrap resamples...", max_entries=8)
def model_uncertainty(game_id, legs_key, legs, B, n_sims, mode, qb_exits, seed):
    from nflsim.parlay import bootstrap_model_uncertainty
    return bootstrap_model_uncertainty(context(), game_id, legs, B, n_sims, seed=seed, mode=mode,
                                       qb_exits=qb_exits, pass_shock=0.0)


def best_config() -> dict:
    p = PROCESSED / "prop_wf_best.json"
    return json.loads(p.read_text()) if p.exists() else {"mode": "independent", "qb_exits": "on"}


def fmt_american(a) -> str:
    return "n/a" if a is None or not np.isfinite(a) else f"{a:+.0f}"


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
lines, props = snapshots()
ctx = context()
glist = A.game_list(lines, props, ctx.games)
best = best_config()
with st.sidebar:
    st.header("Game")
    if glist.empty:
        st.error("No saved odds snapshots. Pull lines with the odds_feed CLI first (costs credits).")
        st.stop()
    gi = st.selectbox("Game", range(len(glist)), format_func=lambda i: glist.loc[i, "label"])
    grow = glist.loc[gi]
    game_id = grow["game_id"]
    book = A.POOLED     # every saved book's prices treated as one book (for now)
    if st.button("Reload snapshots", help="Pick up odds pulled since the app started"):
        snapshots.clear()
        st.rerun()
    st.header("Simulation")
    modes = ["independent", "market_anchored"]
    mode = st.selectbox("Mode", modes, index=modes.index(best.get("mode", "independent")),
                        help="market_anchored matches the market's spread and total; joint structure "
                             "still comes from the simulator.")
    qbx = st.toggle("QB early exits", value=best.get("qb_exits", "on") == "on")
    n_sims = st.select_slider("Simulations", [2000, 5000, 10000, 20000, 50000], value=10000)
    seed = st.number_input("Seed", 0, 10**6, 0)
    settlement = st.selectbox("Push/void settlement", ["reduce", "void_parlay", "push_loses"])
    B = st.slider("Bootstrap refits (model uncertainty)", 0, 20, 10,
                  help="Each refit resamples the drive model and player usage history. 0 = skip.")
    st.caption("Per-game passing shock: **off** (Milestone 2.7 failed its keep test).")
    if best.get("source"):
        st.caption(f"Default mode/QB exits: best-calibrated config in the backtest ({best['source']}).")

def calib_chart(cal: list, cal2: list | None = None, names_=("model", "independence")) -> alt.Chart:
    rows = [{**c, "series": names_[0]} for c in cal] + ([{**c, "series": names_[1]} for c in cal2] if cal2 else [])
    d = pd.DataFrame(rows)
    dom = [names_[0]] + ([names_[1]] if cal2 else [])
    rng = [MODEL_COLOR] + ([INDEP_COLOR] if cal2 else [])
    ref = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
        color=REF_COLOR, strokeDash=[4, 4], strokeWidth=1).encode(x="x", y="y")
    enc = dict(x=alt.X("pred:Q", title="Predicted probability", scale=alt.Scale(domain=[0, 1])),
               y=alt.Y("actual:Q", title="Actual frequency", scale=alt.Scale(domain=[0, 1])),
               color=alt.Color("series:N", scale=alt.Scale(domain=dom, range=rng), title=None,
                               legend=alt.Legend(orient="top")),
               tooltip=[alt.Tooltip("series:N"), alt.Tooltip("bucket:N"), alt.Tooltip("n:Q", format=","),
                        alt.Tooltip("pred:Q", format=".3f"), alt.Tooltip("actual:Q", format=".3f")])
    line = alt.Chart(d).mark_line(strokeWidth=2).encode(**enc)
    pts = alt.Chart(d).mark_circle(size=80, opacity=1, stroke="white", strokeWidth=2).encode(**enc)
    return (ref + line + pts).properties(height=320)


tab_build, tab_log, tab_tail, tab_bt = st.tabs(["Markets & parlay builder", "My parlays", "Tail scanner",
                                                "Backtest results"])

# --------------------------------------------------------------------------- #
# Market browser + parlay builder
# --------------------------------------------------------------------------- #
with tab_build:
    board = A.market_board_all(game_id, lines, props, grow["home_team"], grow["away_team"], show_book=False)
    st.subheader(f"{grow['away_team']} @ {grow['home_team']}, {pd.Timestamp(grow['gameday']):%Y-%m-%d}")
    if board.empty:
        st.warning("No pre-kickoff snapshot for this game.")
    else:
        snap = pd.Timestamp(board["snapshot_time"].min())
        newest = pd.Timestamp(board["snapshot_time"].max())
        age_h = (pd.Timestamp.now(tz="UTC") - snap).total_seconds() / 3600
        when = (f"{snap:%Y-%m-%d %H:%M UTC}" if snap == newest
                else f"{snap:%Y-%m-%d %H:%M} to {newest:%H:%M UTC} (latest per market)")
        src = "all saved books pooled as one (latest pre-kickoff price per market, best price per line)"
        st.caption(f"Prices: {src}, {when}"
                   + (f"; oldest {age_h:.1f} h old" if not grow["played"] else " (game played)"))
        if not grow["played"] and age_h > 6:
            st.warning("Snapshot is more than 6 hours old: prices may have moved.")
    n_alt = int(board["alternate"].sum()) if len(board) else 0
    if n_alt == 0:
        st.info("No alternate lines in the saved snapshots. Pulling them adds the 8 `_alternate` prop "
                f"markets to each game's props call: ≤ {A.alt_cost_per_game()} extra credits per game per "
                f"pull (≤ {A.alt_cost_per_game() * 15} for a 15-game slate). Not pulled: needs your approval "
                "(`python -m nflsim.odds_feed pull-props --alternates --dry-run` shows the plan).")

    sel_key = f"sel|{game_id}|{book}"
    selected = st.session_state.setdefault(sel_key, set())

    def leg_row(i, r):
        c1, c2 = st.columns([5, 1])
        label = r["label"] + ("  ·  alt" if r["alternate"] else "")
        with c1:
            on = st.checkbox(label, key=f"{sel_key}|{i}", value=i in selected, disabled=not r["priceable"],
                             help=None if r["priceable"] else f"Can't price: {r['reason']}")
        with c2:
            st.markdown(f"`{r['price']:+.0f}`" if r["priceable"] else f":gray[{r['price']:+.0f}]")
        (selected.add if on and r["priceable"] else selected.discard)(i)

    left, right = st.columns([3, 2])
    with left:
        if len(board):
            gl = board[board["group"] == "Game lines"]
            if len(gl):
                st.markdown("#### Game lines")
                for i, r in gl.iterrows():
                    leg_row(i, r)
            pl = board[board["group"] != "Game lines"]
            for grp, x in pl.groupby("group", sort=True):
                n_ok = int(x["priceable"].sum())
                with st.expander(f"{grp}  ({n_ok}/{len(x)} priceable)", expanded=bool(x.index.isin(list(selected)).any())):
                    for i, r in x.iterrows():
                        leg_row(i, r)
            st.caption("Greyed out: markets or outcomes the simulator can't price (never guessed). "
                       "Hover for the reason.")

    with right:
        st.markdown("#### Parlay slip")
        chosen = board.loc[sorted(selected)] if selected else board.iloc[0:0]
        if chosen.empty:
            st.write("Click legs on the left to add them.")
        else:
            legs = [replace(r["leg"], label=r["label"]) for _, r in chosen.iterrows()]
            leg_odds = [float(p) for p in chosen["price"]]
            for l, p in zip(legs, leg_odds):
                st.write(f"- {l.label}  `{p:+.0f}`")
            if len({(l.market, l.player_id, l.side, l.line) for l in legs}) < len(legs):
                st.warning("Duplicate leg in the slip.")
            sgp_txt = st.text_input("Book's same-game-parlay price (American, e.g. +450)",
                                    help="The feed has no SGP prices. Enter the price the book shows you.")
            sgp = None
            if sgp_txt.strip():
                try:
                    sgp = float(sgp_txt.replace("+", ""))
                    O.american_to_decimal(sgp)
                except ValueError:
                    st.error("Enter American odds such as +450 or -120.")
                    sgp = None
            run = st.button("Run simulation", type="primary")
            if run:
                st.session_state[f"run|{game_id}"] = True
            if st.session_state.get(f"run|{game_id}"):
                from nflsim.parlay import evaluate
                g = simulated(game_id, n_sims, mode, qbx, int(seed))
                pairs = A.qb_receiver_pairs(legs, g.players.teams)
                if pairs:
                    st.warning("⚠️ " + A.QB_WR_WARNING + "  Pairs: " +
                               "; ".join(f"{q.label} + {r.label}" for q, r in pairs))
                mu = None
                if B > 0:
                    lk = tuple((l.market, l.side, l.line, l.player_id) for l in legs)
                    mu = model_uncertainty(game_id, lk, legs, B, 2000, mode, qbx, int(seed) + 1)
                res = evaluate(g, legs, settlement, leg_odds, sgp, mu)
                j = res.joint
                rows = []
                for l, r, p in zip(legs, res.leg_rates, leg_odds):
                    dec = float(O.american_to_decimal(p))
                    ev = O.expected_value(r["p_win"], dec, r["p_push"] + r["p_void"])["roi"]
                    fair = O.fair_odds(r["p_win_given_decided"])["american"] \
                        if 0 < r["p_win_given_decided"] < 1 else np.nan
                    rows.append({"leg": l.label, "P(win)": r["p_win"], "P(push)": r["p_push"],
                                 "P(void)": r["p_void"], "fair (decided)": fmt_american(fair),
                                 "offered": f"{p:+.0f}", "leg EV/unit": ev})
                st.dataframe(pd.DataFrame(rows).style.format(
                    {"P(win)": "{:.3f}", "P(push)": "{:.3f}", "P(void)": "{:.3f}", "leg EV/unit": "{:+.3f}"}),
                    hide_index=True, width="stretch")
                c1, c2, c3 = st.columns(3)
                c1.metric("Joint P (K/N)", f"{j['p']:.4f}", help=f"{j['K']} of {j['N']} simulations")
                c2.metric("Independence benchmark", f"{res.independence_benchmark:.4f}",
                          help="Product of leg win rates: comparison only, never the price.")
                c3.metric("Fair odds", fmt_american(res.fair["american"]) if res.fair else "n/a")
                st.caption(f"95% Monte Carlo CI for the joint probability: [{j['ci95'][0]:.4f}, {j['ci95'][1]:.4f}]")
                if res.offered and res.ev and sgp is not None:
                    c1, c2 = st.columns(2)
                    c1.metric("Offered (your SGP price)", f"{sgp:+.0f}",
                              help=f"implied {res.offered['implied']:.4f}")
                    c2.metric("Expected ROI per unit", f"{res.ev['roi']:+.3f}",
                              help=f"Monte Carlo SE {res.ev['mc_se']:.3f}; settlement '{settlement}'")
                elif sgp is None:
                    st.caption("Enter the book's SGP price to see expected ROI.")
                for n_ in res.notes:
                    st.caption("Note: " + n_)
                st.markdown("**Uncertainty**")
                st.write(f"- Monte Carlo (finite simulations): SE {j['mc_se']:.4f}. Shrinks with more simulations.")
                if mu:
                    st.write(f"- Model (bootstrap, {mu['B']} refits × {mu['n_sims']} sims): sd {mu['model_sd']:.4f} "
                             f"after removing Monte Carlo noise; 5–95% range [{mu['q05']:.4f}, {mu['q95']:.4f}]. "
                             "**More simulations do not reduce this.**")
                else:
                    st.write("- Model uncertainty: not computed (set bootstrap refits > 0).")
                if len(legs) > 1:
                    st.markdown("**Pairwise correlation of leg wins**")
                    C = pd.DataFrame(res.corr, index=[l.label for l in legs], columns=[f"L{i+1}" for i in range(len(legs))])
                    st.dataframe(C.style.format("{:+.2f}", na_rep="–"), width="stretch")
                with st.expander("Missing data and assumptions used"):
                    for m in res.missing:
                        st.write("- " + m)
                if st.button("Save to My parlays"):
                    A.log_parlay({"logged_at": datetime.now(timezone.utc).isoformat(), "game_id": game_id,
                                  "game": f"{grow['away_team']} @ {grow['home_team']}", "book": book,
                                  "legs": [A.leg_to_dict(l) for l in legs], "leg_odds": leg_odds,
                                  "parlay_odds": sgp, "settlement": settlement, "mode": mode,
                                  "qb_exits": qbx, "n_sims": n_sims, "p_joint": j["p"], "ci95": j["ci95"],
                                  "p_indep": res.independence_benchmark,
                                  "fair_american": res.fair["american"] if res.fair else None,
                                  "ev_roi": res.ev["roi"] if res.ev and sgp is not None else None,
                                  "qb_receiver_warning": bool(pairs), "result": None})
                    st.success("Saved. Grade it in 'My parlays' after the game.")
                if grow["played"]:
                    gr = A.grade_parlay_actual(legs, game_id, leg_odds, sgp, settlement)
                    if gr is None:
                        st.info("Official player stats for this game aren't in the local data yet "
                                "(run `python -m nflsim.ingest`).")
                    else:
                        st.markdown(f"**Actual result** ({gr['final']}): parlay **{gr['status']}**; legs: "
                                    + ", ".join(f"{l.label} → {x}" for l, x in zip(legs, gr["legs"])))

# --------------------------------------------------------------------------- #
# Parlay log
# --------------------------------------------------------------------------- #
with tab_log:
    st.subheader("My parlays")
    if st.button("Grade finished games"):
        A.grade_log()
        st.success("Graded every logged parlay whose game has official results.")
    log = A.load_log()
    if not log:
        st.write("No parlays logged yet.")
    else:
        rows = []
        for e in log:
            r = e.get("result") or {}
            pay = r.get("payout_per_unit")
            rows.append({"logged": e["logged_at"][:16], "game": e["game"], "book": e["book"],
                         "legs": " + ".join(d.get("label") or d["market"] for d in e["legs"]),
                         "SGP price": fmt_american(e.get("parlay_odds")), "model P": e["p_joint"],
                         "fair": fmt_american(e.get("fair_american")), "exp. ROI": e.get("ev_roi"),
                         "QB+WR warning": "⚠️" if e.get("qb_receiver_warning") else "",
                         "status": r.get("status", "pending"),
                         "profit/unit": None if pay is None else pay - 1})
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, width="stretch")
        done = df[df["profit/unit"].notna()]
        if len(done):
            st.write(f"Settled: {len(done)} parlays, total profit {done['profit/unit'].sum():+.2f} units "
                     f"(flat 1-unit stakes). A small sample says little about skill.")

# --------------------------------------------------------------------------- #
# Tail scanner page = game report (Milestone 4): saved reports + backtest trust labels
# --------------------------------------------------------------------------- #
GREY = "color: #8a8984"


def _pct(p) -> str:
    return "" if p is None or pd.isna(p) else f"{100 * p:.1f}%"


def _rng(lo, hi) -> str:
    return "" if pd.isna(lo) or pd.isna(hi) else f"{100 * lo:.1f}–{100 * hi:.1f}%"


def _greyed(df: pd.DataFrame, mask: pd.Series, fmt: dict | None = None):
    def _style(row):
        return [GREY if mask.loc[row.name] else "" for _ in row]
    return df.style.apply(_style, axis=1).format(fmt or {}, na_rep="")


with tab_tail:
    from nflsim import gamereport as GR
    from nflsim import tails as TL
    st.subheader("Game report: everything the simulation says happens")
    trep = TL.load_report()
    if trep is None:
        st.warning("No game-report backtest yet (`python -m nflsim.tails backtest`): nothing here is validated.")
    else:
        kt = trep["keep_test"]
        (st.success if kt["overall"] == "PASS" else st.error)(
            f"Backtest keep test (2024 → 2026 wk 7): **{kt['overall']}**. Calibration by level: "
            + ", ".join(f"{k} {v}" for k, v in kt["a_calibration_by_level"].items())
            + f"; percentile check {kt['b_percentiles_pooled']}. Untrusted stat/levels are greyed, with the reason.")
    saved = GR.saved_reports()
    c1, c2 = st.columns([3, 2])
    with c2:
        if st.button(f"Build report for {grow['away_team']} @ {grow['home_team']}",
                     help="10,000 simulations + 10 bootstrap refits; saved odds only (no API calls). ~30 s."):
            with st.spinner("Simulating and refitting..."):
                cfg_ = TL.load_config()
                base_ = TL.Baseline(TL.load_actuals(), context().player_games, context().roles, cfg_["thresholds"], cfg_)
                GR.export(GR.build(context(), base_, cfg_, game_id, names=names(), rep=trep, verbose=False))
            saved = GR.saved_reports()
    if not saved:
        st.write("No saved reports. Run `python -m nflsim.gamereport build` (upcoming week) or use the button.")
    else:
        with c1:
            rid = st.selectbox("Game", saved, index=saved.index(game_id) if game_id in saved else 0)
        R = GR.load(rid)
        m = R["meta"]
        a_ = m.get("anchor") or {}
        st.caption(
            f"**{m['matchup']}** · kickoff {str(m['kickoff_utc'])[:16]} UTC · built {m['built_at'][:16]} UTC · "
            f"{m['n_sims']:,} sims · model range from {m['bootstrap']} bootstrap refits × {m['boot_sims']:,} · "
            + (f"market-anchored (home margin {a_.get('target_margin')}, total {a_.get('target_total')})"
               if a_.get("mode") == "market_anchored" else f"independent ({a_.get('fallback', '')})")
            + f" · QB exits {'on' if m['qb_exits'] else 'off'} · pass shock off")
        if a_.get("warning"):
            st.warning(a_["warning"])
        if a_.get("mode") == "market_anchored":
            st.caption("Market-anchored: the mean margin and total are set to the market's; the spread of outcomes, "
                       "drives, plays and every player number come from the simulator.")

        st.markdown("#### Game")
        tp_ = R["team_probs"]
        g1, g2 = st.columns([2, 3])
        g1.dataframe(pd.DataFrame({"outcome": tp_["outcome"], "sim": tp_["p"].map(_pct),
                                   "MC 95%": [_rng(a, b) for a, b in zip(tp_["mc_lo"], tp_["mc_hi"])],
                                   "model range": [_rng(a, b) for a, b in zip(tp_["model_lo"], tp_["model_hi"])]}),
                     hide_index=True, width="stretch")
        if m.get("team_range_note"):
            g1.caption(m["team_range_note"])
        g2.dataframe(R["team_dist"].rename(columns={"p10": "10th", "p25": "25th", "p50": "median", "p75": "75th",
                                                    "p90": "90th"}).round(2), hide_index=True, width="stretch")
        with st.expander("Team inputs and opponent context"):
            st.dataframe(pd.DataFrame(m["opponent_context"]).round(3), hide_index=True, width="stretch")
            st.caption(m["opponent_note"] + " EPA/success allowed are opponent-adjusted per-play rates "
                       "(higher = weaker defense).")

        st.markdown("#### What moves together")
        cr = R["correlations"]
        show_all = st.toggle("Show every pair (incl. same-player and game-level)", value=False)
        cv = cr if show_all else cr[cr["top"] != ""].sort_values("corr", ascending=False)
        st.dataframe(cv[["leg_a", "leg_b", "corr", "kind"]].round(3), hide_index=True, width="stretch",
                     height=320 if show_all else "content")
        st.caption("Pearson correlation across simulations; a player's value is 0 in simulations where he sits.")

        st.markdown("#### Players")
        thr, sts = R["player_thresholds"], R["player_stats"]
        f1, f2, f3, f4, f5 = st.columns(5)
        teams_ = f1.multiselect("Team", [m["away"], m["home"]], default=[m["away"], m["home"]])
        pos_ = f2.multiselect("Position", sorted(thr["position"].unique()), default=sorted(thr["position"].unique()))
        lv_ = f3.multiselect("Level", list(TL.LEVELS), default=list(TL.LEVELS))
        stat_opts = [s for s in TL.STAT_COLS if s in set(thr["stat"])]
        stats_ = f4.multiselect("Stat", stat_opts, default=stat_opts, format_func=TL.STAT_LABELS.get)
        sort_ = f5.selectbox("Sort by", ["lift", "hit %", "threshold height"])
        t1, t2 = st.columns(2)
        trusted_only = t1.toggle("Trusted only", value=False)
        min_active = t2.slider("Min P(active)", 0, 100, 0, 5, help="Hit % is conditional on playing.")
        pa = sts.groupby("player_id")["p_active"].first()
        d = thr[thr["team"].isin(teams_) & thr["position"].isin(pos_) & thr["level"].isin(lv_)
                & thr["stat"].isin(stats_) & (thr["player_id"].map(pa) >= min_active / 100)]
        if trusted_only:
            d = d[d["trust"] == "trusted"]
        if sort_ == "lift":
            d = d.sort_values("lift", ascending=False)
        elif sort_ == "hit %":
            d = d.sort_values("p_sim", ascending=False)
        else:
            d = d.sort_values(["stat", "at_least", "p_sim"], ascending=[True, False, False])
        view = pd.DataFrame({
            "player": d["name"], "team": d["team"], "role": d["role"],
            "outcome": [f"{TL.STAT_LABELS[s]} ≥ {t}" for s, t in zip(d["stat"], d["at_least"])], "level": d["level"],
            "sim %": 100 * d["p_sim"], "MC 95%": [_rng(a, b) for a, b in zip(d["mc_lo"], d["mc_hi"])],
            "model range": [_rng(a, b) for a, b in zip(d["model_lo"], d["model_hi"])],
            "role %": 100 * d["p_role"], "own %": 100 * d["p_hist"], "games": d["n_hist_games"],
            "lift": 100 * d["lift"], "market % (info)": 100 * d["p_market"].astype(float),
            "active %": 100 * d["player_id"].map(pa), "trust": d["trust"]}).reset_index(drop=True)
        bad = ~view["trust"].eq("trusted")
        st.dataframe(_greyed(view, bad, {"sim %": "{:.1f}", "role %": "{:.1f}", "own %": "{:.1f}", "lift": "{:+.1f}",
                                         "market % (info)": "{:.1f}", "active %": "{:.0f}"}),
                     hide_index=True, width="stretch", height=480)
        st.caption(f"{len(view):,} player-thresholds. Hit % is over simulations where the player is active. "
                   f"Role % = players in the same pregame role, previous {m['role_prior_seasons']} seasons; own % = "
                   f"his last {m['history_window_games']} games played, shrunk toward the role with "
                   f"{m['shrink_prior_games']:g} games of weight; lift = sim − role. MC 95% shrinks with more "
                   "simulations, the model range does not. Greyed = the backtest says not to trust that stat/level.")

        hist, use, ml = R["player_history"], R["player_usage"], R["market_lines"]
        for team in teams_:
            st.markdown(f"##### {team}")
            order = sts[sts["team"] == team].groupby("player_id")["p_active"].first().sort_values(ascending=False)
            for pid, pact in order.items():
                if pact < min_active / 100:
                    continue
                x = sts[sts["player_id"] == pid]
                r0 = x.iloc[0]
                usage_ = r0["usage_role"] if isinstance(r0["usage_role"], str) and r0["usage_role"] else ""
                with st.expander(f"{r0['name']} · {r0['position']} · role {r0['role']}"
                                 + (f" (usage {usage_})" if usage_ and usage_ != r0["role"] else "")
                                 + f" · active {100 * pact:.0f}%"):
                    st.markdown(f"**Drivers:** {r0['drivers']}")
                    pv = x[["stat", "mean", "p10", "p25", "p50", "p75", "p90"]].copy()
                    pv["stat"] = pv["stat"].map(TL.STAT_LABELS)
                    st.dataframe(pv.rename(columns={"p10": "10th", "p25": "25th", "p50": "median", "p75": "75th",
                                                    "p90": "90th"}).round(1), hide_index=True, width="stretch")
                    e1, e2 = st.columns(2)
                    with e1:
                        st.markdown("**Model inputs** (vs the average each is shrunk toward)")
                        for t_ in use.loc[use["player_id"] == pid, "text"]:
                            st.markdown(f"- {t_}")
                    with e2:
                        mx = ml[ml["player_id"] == pid]
                        if len(mx):
                            st.markdown("**Saved market lines** (info only)")
                            st.dataframe(pd.DataFrame({"market": mx["market"], "line": mx["line"],
                                                       "market %": mx["p_market"].map(_pct),
                                                       "sim %": mx["p_sim_over"].map(_pct), "books": mx["n_books"]}),
                                         hide_index=True, width="stretch")
                    hx = hist[hist["player_id"] == pid]
                    st.markdown(f"**Game log** — last {len(hx)} games played (window {m['history_window_games']})")
                    cols_ = [s for s in TL.STAT_COLS if s in set(x["stat"])]
                    st.dataframe(hx[["gameday", "opponent", "role_then"] + cols_].rename(columns=TL.STAT_LABELS),
                                 hide_index=True, width="stretch")
                    tx = thr[thr["player_id"] == pid]
                    tv = pd.DataFrame({"outcome": [f"{TL.STAT_LABELS[s]} ≥ {t}" for s, t in zip(tx["stat"], tx["at_least"])],
                                       "level": tx["level"], "sim %": 100 * tx["p_sim"],
                                       "MC 95%": [_rng(a, b) for a, b in zip(tx["mc_lo"], tx["mc_hi"])],
                                       "model range": [_rng(a, b) for a, b in zip(tx["model_lo"], tx["model_hi"])],
                                       "role %": 100 * tx["p_role"], "own %": 100 * tx["p_hist"],
                                       "lift": 100 * tx["lift"], "trust": tx["trust"]}).reset_index(drop=True)
                    st.markdown("**Every threshold**")
                    st.dataframe(_greyed(tv, ~tv["trust"].eq("trusted"),
                                         {"sim %": "{:.1f}", "role %": "{:.1f}", "own %": "{:.1f}", "lift": "{:+.1f}"}),
                                 hide_index=True, width="stretch", height=300)
        st.caption(f"Exported: `reports/{rid}/` (CSV tables + report.html).")

        if trep is not None:
            with st.expander("Backtest: calibration and percentile check (2024 → 2026 wk 7)"):
                st.write(f"{trep['n_graded']:,} graded player-thresholds in {trep['n_games']} games; "
                         f"{trep['n_stat_rows_graded']:,} player-stat results. Bands with n < 200 are not scored.")
                rows_ = []
                for s_, r_ in trep["by_stat"].items():
                    pc = r_["percentile"]
                    rows_.append({"stat": TL.STAT_LABELS[s_], "n": r_["n"],
                                  "max band err (pts)": None if r_["max_abs_err_scored"] is None
                                  else 100 * r_["max_abs_err_scored"],
                                  "calibration": {True: "PASS", False: "FAIL", None: "no data"}[r_["pass"]],
                                  "below p10 %": None if not pc.get("n") else 100 * pc["below_p10"],
                                  "below median %": None if not pc.get("n") else 100 * pc["below_p50"],
                                  "above p90 %": None if not pc.get("n") else 100 * pc["above_p90"],
                                  "percentiles": {True: "PASS", False: "FAIL", None: "no data"}[pc.get("pass")],
                                  **{lv: trep["trust"][s_].get(lv, "") for lv in TL.LEVELS}})
                st.dataframe(pd.DataFrame(rows_).round(1), hide_index=True, width="stretch")
                lv_sel = st.selectbox("Calibration bands for", ["all levels"] + list(trep["by_level"]) +
                                      [TL.STAT_LABELS[s_] for s_ in trep["by_stat"]])
                inv = {v: k for k, v in TL.STAT_LABELS.items()}
                src_ = trep["by_level"][lv_sel] if lv_sel in trep["by_level"] else \
                    trep["by_stat"][inv[lv_sel]] if lv_sel in inv else None
                if src_ is None:
                    for lv, c in trep["by_level"].items():
                        st.markdown(f"**{lv}** — {'PASS' if c['pass'] else 'FAIL'}")
                        st.altair_chart(calib_chart([{**b, "bucket": b["band"]} for b in c["bands"]],
                                                    names_=(f"{lv} thresholds", "")), width="stretch")
                else:
                    b1, b2 = st.columns([3, 2])
                    b1.altair_chart(calib_chart([{**b, "bucket": b["band"]} for b in src_["bands"]],
                                                names_=(lv_sel, "")), width="stretch")
                    bt = pd.DataFrame(src_["bands"])
                    if "actual_ci95" in bt:
                        bt["actual 95% CI"] = [_rng(*c) if isinstance(c, list) else "" for c in bt["actual_ci95"]]
                    b2.dataframe(bt.drop(columns=[c for c in ("actual_ci95",) if c in bt]).round(4),
                                 hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
# Backtest results
# --------------------------------------------------------------------------- #
CONFIGS = [("independent", "on", ""), ("independent", "off", "_noexits"),
           ("market_anchored", "on", "_anchored"), ("market_anchored", "off", "_anchored_noexits")]


with tab_bt:
    st.subheader("Walk-forward backtest (weekly refits, pass shock off)")
    wf = PROCESSED / "wf_report.json"
    if wf.exists():
        e = json.loads(wf.read_text())["evaluation"]
        st.markdown("**Game level, independent mode, 2021 → latest** "
                    "(market-anchored game metrics are circular and not shown as skill)")
        st.dataframe(pd.DataFrame({k: e["wp"][k] for k in ("model_sim", "elo", "home_baseline", "market_close")}).T,
                     width="stretch")
        st.altair_chart(calib_chart([{"bucket": k, **v} for k, v in e["model_calibration"].items()],
                                    names_=("model win probability", "")), width="stretch")
    reps = {}
    for m, q, sx in CONFIGS:
        p = PROCESSED / f"prop_wf_report{sx}.json"
        if p.exists():
            reps[f"{m} / exits {q}"] = json.loads(p.read_text())
    if not reps:
        st.write("Run `python -m nflsim.backtest --props` (each config) to fill this page.")
    else:
        st.markdown("**Configurations** (props at self-set lines and synthetic SGPs)")
        cmp_ = pd.DataFrame([{"config": k, "props n": r["props_overall"]["n"],
                              "props Brier": r["props_overall"]["brier"],
                              "props log loss": r["props_overall"]["log_loss"],
                              "SGP n": r["sgp_overall"]["joint"]["n"],
                              "SGP Brier (K/N)": r["sgp_overall"]["joint"]["brier"],
                              "SGP log loss (K/N)": r["sgp_overall"]["joint"]["log_loss"],
                              "SGP log loss (independence)": r["sgp_overall"]["independence"]["log_loss"]}
                             for k, r in reps.items()])
        st.dataframe(cmp_, hide_index=True, width="stretch")
        cfg = st.selectbox("Config", list(reps), index=list(reps).index(
            f"{best.get('mode', 'independent')} / exits {best.get('qb_exits', 'on')}")
            if f"{best.get('mode', 'independent')} / exits {best.get('qb_exits', 'on')}" in reps else 0)
        r = reps[cfg]
        mk = st.selectbox("Prop market / line type", sorted(r["props_by_market"]))
        x = r["props_by_market"][mk]
        c1, c2 = st.columns([3, 2])
        with c1:
            st.altair_chart(calib_chart(x["calibration"], names_=(mk, "")), width="stretch")
        with c2:
            st.write(f"n = {x['n']:,}; Brier {x['brier']:.4f} (in-sample base rate {x['brier_insample_base_rate']:.4f}, "
                     "a reference, not a forecast); log loss {x['log_loss']:.4f}")
            st.dataframe(pd.DataFrame(x["calibration"]), hide_index=True, width="stretch")
        st.markdown("**Synthetic same-game parlays: K/N vs independence benchmark**")
        c1, c2 = st.columns([3, 2])
        with c1:
            st.altair_chart(calib_chart(r["sgp_calibration_joint"], r["sgp_calibration_indep"],
                                        ("K/N (shared simulation)", "independence")), width="stretch")
        with c2:
            t = pd.DataFrame(r["sgp_by_template"]).T[["n", "mean_p_joint", "mean_p_indep", "actual_rate",
                                                      "brier_joint", "brier_indep"]]
            st.dataframe(t.astype(float).round(4), width="stretch")
            st.caption(r["sgp_note"])
        if "real_lines" in r:
            rl = r["real_lines"]
            st.markdown(f"**Real saved prices** ({rl['n_games']} games; {rl['note']})")
            cal = rl["calibration_two_sided"]
            st.write(f"Over side, two-sided markets, decided: n = {cal['n']:,} book-lines in {cal['n_games']} games. "
                     f"Brier model {cal['model']['brier']:.4f} vs market (de-vigged) {cal['market_devigged']['brier']:.4f}.")
            st.dataframe(pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk != "by_market"}
                                       for k, v in rl["roi"].items()}).T, width="stretch")
