# nfl-sim

A Monte Carlo simulator for NFL games, built from pregame information. See `../SPEC.md` for the full plan.
**Status: Milestones 1 (data, team model, drive-level simulator) and 2 (players, markets, parlays, odds) are done, plus a market-anchored simulation mode and a cleanup round (template-leakage check, QB early exits, close-game endings).** Milestone 3 (full backtest with prop calibration, Streamlit app) is not started.

## Install
```bash
cd nfl-sim
python3 -m venv .venv            # Python 3.11+ (built and tested on 3.13)
.venv/bin/pip install -e '.[dev]'
```

## Run
```bash
.venv/bin/python -m nflsim.ingest          # download/cache nflverse 2016..current (re-runs fetch only missing + current season)
.venv/bin/python -m nflsim.features        # build processed tables + as-of game features (~10 s)
.venv/bin/python -m pytest -q              # tests (need the two steps above)
.venv/bin/python -m nflsim.backtest --start 2021 --sims 4000   # weekly walk-forward (~3 min)
.venv/bin/python -m nflsim.backtest --reuse                     # re-score saved predictions
.venv/bin/python -m nflsim.simulator --game-id 2026_07_DAL_PHI --sims 10000   # one matchup + drive log
.venv/bin/python -m nflsim.players         # build play templates, player usage, pregame rosters (~10 s)
.venv/bin/python -m nflsim.backtest --players --player-seasons 2024 2025 --sims 1000   # M2 player validation (~10 min)
.venv/bin/python scripts/estimate_dispersion.py 2023   # re-estimate share dispersion (2023 only)
.venv/bin/python -m nflsim.parlay --game-id 2024_01_BAL_KC \
    --leg "passing_yards:Patrick Mahomes:over:249.5" --leg "receiving_yards:Rashee Rice:over:55.5" \
    --leg "moneyline:KC" --leg "total:over:46.5" \
    [--leg-odds A B C D] [--parlay-odds X] [--settlement reduce|void_parlay|push_loses] --sims 20000 \
    [--mode independent|market_anchored] [--spread -3.5] [--total 46.5] [--qb-exits on|off]
.venv/bin/python -m nflsim.backtest --mode market_anchored --sims 4000                  # anchored game walk-forward
.venv/bin/python -m nflsim.backtest --players --mode market_anchored --player-seasons 2024 2025 --sims 1000
.venv/bin/python scripts/compare_modes.py          # independent vs anchored report
.venv/bin/python scripts/check_player_anchor.py    # tolerance check for the 1,000-sim player run
.venv/bin/python scripts/qb_exits.py [seasons]     # starter-exit rates/hazards (default 2018-2020)
.venv/bin/python scripts/blowout_starters.py       # other starters' usage after a QB exit (2018-2020)
.venv/bin/python scripts/close_games.py 200        # close-game diagnosis, actual vs simulated (2018-2020)
.venv/bin/python scripts/tune_window.py <tag>      # 2018-2020 walk-forward (tuning window only)
.venv/bin/python scripts/cleanup_report.py before_qbexit exitfinal_coarse .   # before/after tables
```
`NFLSIM_COARSE_LATE=1` reproduces the old (pre-cleanup) late-game states for before/after comparisons.
`--qb-exits on|off` (default `on`) switches simulated starting-QB early exits in the parlay CLI and in both backtests (game and `--players`), in either mode. Outputs from `off` runs get a `_noexits` suffix. In Python: `simulate_game(..., qb_exits=False)`. Milestone 3 will compare prop/parlay calibration both ways.
`python scripts/exit_corr_split.py` splits QB correlations by exit vs no-exit games (needs the independent `--players` run).
Run order for a fresh checkout: `ingest` → `features` → `players` → tests. Results go to `data/processed/` (`wf_*` for games, `player_wf_*` for players).
Leg syntax: `<market>:<player name or gsis_id>:<over|under>:<line>`, `anytime_td:<player>:<yes|no>`, `spread:<TEAM>:<line>`, `moneyline:<TEAM>`, `total:<over|under>:<line>`. Player markets: passing_yards, passing_tds, pass_attempts, completions, rushing_yards, rush_attempts, receptions, receiving_yards, anytime_td.

## What's in it
| module | does |
|---|---|
| `ingest.py` | Downloads schedules, pbp, weekly player stats, rosters, snap counts, depth charts and injuries into `data/raw/<dataset>/<season>.parquet`. Normalizes team codes (OAK→LV, SD→LAC, STL/LAR→LA, JAC→JAX). Players are keyed by `gsis_id`. Keeps a manifest with download times and errors. |
| `features.py` | Builds a drive table. Builds team off/def EPA/play and success rate, opponent-adjusted (each past game is adjusted by the opponent's *pregame* rating), recency-weighted (8-game half-life, ×0.5 per season boundary) and shrunk toward a trailing-year league average (400-play prior). Also adds pregame starting QB plus that QB's shrunk EPA/dropback and `qb_delta` (expected starter minus the dropback-weighted mix of QBs behind the team's offense rating), rest days, home/neutral, and `asof_date`. |
| `team_model.py` | Elo with margin-of-victory multiplier and data-estimated home-field advantage. A multinomial logistic drive-outcome model over TD / FG / missed FG / punt / turnover / downs / end of half / safety / defensive TD, conditioned on the matchup, field position, clock and score state. Empirical pace, field-position and PAT/2-pt tables from pbp. `expected_points_per_drive()`. |
| `simulator.py` | Vectorized drive-level simulator with kickoffs, possession changes, clock by half, points by quarter, overtime rules by season (10-min OT, guaranteed possession in playoffs ≥2022 and regular season ≥2025, regular-season ties), and an optional drive log. |
| `players.py` | Turns simulated drives into player box scores (see "Milestone 2" below). |
| `markets.py` | Leg definitions. Grades each leg in each simulation as win / loss / push / void. |
| `parlay.py` | Joint K/N with Wilson CI, independence benchmark, pairwise leg correlations, settlement rules, EV, bootstrap model uncertainty, CLI. |
| `odds.py` | American ↔ decimal ↔ probability, two-way de-vig (multiplicative, power), fair odds, EV/ROI. |
| `pipeline.py` | Loads all inputs once; `simulate_game()` runs the team and player layers for one game, in either mode. |
| `anchor.py` | Market-anchored mode: market line lookup, the two-parameter solver, and its report. |
| `backtest.py` | Weekly walk-forward. For each week it refits on data dated before the week's first game, then simulates every game. Compares the model against a home-team baseline, Elo and the closing market. Also checks league-wide distributions. |

## Validation (walk-forward, 2021 → 2026 week 7, 1,530 games, 4,000 sims per game)
Win probability (5 ties excluded, n = 1,525):

| | Brier | Log loss | Accuracy |
|---|---|---|---|
| **Model (simulation)** | 0.2214 | 0.6332 | 62.9% |
| Elo alone | 0.2251 | 0.6417 | 63.7% |
| Home-team baseline | 0.2484 | 0.6899 | 54.2% |
| Closing market (de-vigged ML) | **0.2131** | **0.6140** | 66.3% |

Score error (n = 1,530):

| | Margin MAE | Margin RMSE | Total MAE | Total RMSE | Total bias (actual − pred) |
|---|---|---|---|---|---|
| Model | 10.08 | 12.99 | 10.68 | 13.37 | −0.84 |
| Home baseline | 10.92 | 14.09 | 10.96 | 13.74 | −0.77 |
| Closing market | 9.74 | 12.61 | 10.33 | 13.07 | +0.55 |

Model team-points MAE: 7.44. **The model is clearly worse than the closing market** on every metric. It beats the home-team baseline, and beats Elo on Brier and log loss.

League-wide, simulated vs actual (same games):

| | Simulated | Actual |
|---|---|---|
| Points per team-game | 23.0 | 22.6 |
| Total points mean / sd | 46.1 / 12.8 | 45.2 / 13.7 |
| Drives per game | 21.1 | 21.7 |
| Home win rate | 54.0% | 54.3% |
| Margin sd | 13.9 | 14.1 |
| Final margin = 3 / = 7 | 13.1% / 8.3% | 15.6% / 8.0% |
| One-score games (≤7) | 45.5% | 49.8% |
| Overtime | 6.0% | 6.3% |
| Ties | 0.65% | 0.33% |
| Model favorite: predicted / actual win rate | 63.8% / 62.9% | |
| Market favorite: implied / actual win rate | 66.0% / 66.3% | |

Tuning was done only on a 2018–2020 walk-forward (validation). The grid covered rating half-life ∈ {5, 8, 12}, shrinkage ∈ {200, 400, 800} and drive-model season decay ∈ {1.0, 0.7, 0.5}. The 2021+ results above are out of sample.

### QB-delta fix (applied after the first M1 report)
`qb_delta` originally compared the expected starter with **last game's** QB. A backup making his 2nd+ start therefore got no adjustment, even though the team's offense rating was still mostly built from the injured starter's games. Now it compares with the dropback-weighted mix of QBs in the games behind the offense rating (same recency weights). No tuned settings were changed. Same walk-forward, seed 0, 4,000 sims:

| Games | n | Brier before → after | Log loss before → after | Margin MAE before → after | Total MAE before → after | Market Brier |
|---|---|---|---|---|---|---|
| All | 1,530 (1,525 for WP) | 0.2216 → 0.2212 | 0.6337 → 0.6327 | 10.105 → 10.082 | 10.749 → 10.765 | 0.2131 |
| Starter ≠ last game's QB | 327 | 0.2218 → 0.2209 | 0.6351 → 0.6331 | 10.285 → 10.225 | 10.558 → 10.667 | 0.2133 |
| Starter < 50% of rating's QB mix | 677 | 0.2199 → 0.2200 | 0.6289 → 0.6292 | 10.277 → 10.244 | 10.757 → 10.784 | 0.2123 |
| Starter < 8 prior games | 252 | 0.2111 → 0.2097 | 0.6118 → 0.6085 | 10.194 → 10.098 | 10.626 → 10.618 | 0.2101 |
| Any of the above | 788 | 0.2194 → 0.2191 | 0.6291 → 0.6283 | 10.245 → 10.194 | 10.651 → 10.718 | 0.2131 |

All overall changes are within noise. Per-game paired bootstrap 95% CIs: Brier −0.0014 to +0.0005, total MAE −0.017 to +0.050, margin MAE −0.054 to +0.009. Changing the simulation seed alone moves total MAE by 0.009 and Brier by 0.0003. The fix was kept because nothing got measurably worse and it fixes a real logic error. It is not a demonstrated accuracy gain.

These tables are the current code (independent mode). They include the QB-delta fix, the field-position-aware pace table (Milestone 2), per-simulation common random numbers (added for market anchoring), and the cleanup round's QB early exits and late-game score-situation states (see "Cleanup round"). The before/after tables in the fix sections were each measured at the time of that fix.

### Pace table change (Milestone 2)
To give each simulated drive a real play-by-play template, the pace table now samples historical drives by outcome × game state × **starting field-position bucket** (20-yard buckets, with fallbacks for small cells), not just outcome × state. Same walk-forward, seed 0: Brier 0.2212 → 0.2212, log loss 0.6327 → 0.6329, margin MAE 10.082 → 10.082, total MAE 10.765 → 10.727 (paired-bootstrap 95% CI on the Brier change −0.0005 to +0.0005). Kept: neutral on win probability and slightly better on totals.

## Milestone 2: players, markets, parlays, odds
**How player stats are made.** Each simulated drive reuses a real historical drive (same outcome, game state and field-position bucket) as its play-by-play template: play types (targeted pass, throwaway/spike, sack, designed run, scramble, kneel), completions, interceptions, yards and the TD play. Yards are rescaled so a TD drive gains exactly the simulated distance to the goal line, and a non-scoring drive never reaches it. Each play is then given to a player:
- dropbacks, scrambles, kneels → the expected starting QB
- targeted passes → receiver with P ∝ target share × P(catch) × P(yards bin) × TD ratio (Bayes rule given the template play)
- designed runs → carrier with P ∝ carry share × P(yards bin) × TD ratio

Shares and efficiency come from each player's recent games (6-game half-life, ×0.5 per season), shrunk toward the average for his pregame role (position + depth-chart rank, e.g. WR2, RB1).

**Availability.** Players listed Out or Doubtful on the final injury report get zero, and their shares go to teammates by re-normalization. Everyone else has P(active): the share of the team's last 4 games this season he played, shrunk toward his role's base rate. In each simulation he is active or not; when he sits, his share is redistributed and his props are VOID in that simulation (books void props on players who don't play). The expected starting QB is always active.

**Game-to-game share variation.** Per simulation, target and carry shares are drawn from a Dirichlet with concentration α. α was estimated by method of moments on **2023 only**: α_targets = 63.5 (1.5× binomial variance at 35 targets), α_carries = 11.35 (3.8×). The excess variance also contains share-model error, which is intended: the predictive spread then matches what the model actually misses.

**Consistency (rule 3), tested on 9,000 simulated games in each mode:** receptions ≤ targets; team completions = Σ receptions = starter + backup QB completions; Σ receiving yards = QB passing yards; Σ receiving TDs = QB passing TDs; player offensive TDs = TD plays; only the starter (and, after a simulated exit, the backup) throws; inactive players record nothing; team points = 6·TDs + 3·FGs + 2·safeties + PATs.

### Player validation (weekly walk-forward, 2024–2025 seasons, 1,000 sims per game)
*Measured before the cleanup round (no QB early exits). Current QB1 numbers and correlations are in "Cleanup round" below; receiver, TE and RB rows moved by at most 1 unit or yard (mean, sd, p10, p90).*

Roles are the model's own pregame roles (QB1 = expected starter; WR1–3/TE1 by target share; RB1–2 by carry share). Actuals are official nflverse weekly stats for players who played. Simulated samples are from simulations where the player was active.

| Role | Stat | n | Actual mean / sd | Simulated mean / sd | Actual p10–p90 | Simulated p10–p90 |
|---|---|---|---|---|---|---|
| QB1 | pass attempts | 1,063 | 31.0 / 9.1 | 33.0 / 7.1 | 20–42 | 24–42 |
| QB1 | completions | 1,063 | 20.2 / 6.3 | 21.6 / 5.0 | 13–28 | 15–28 |
| QB1 | passing yards | 1,063 | 222.5 / 77.8 | 234.7 / 64.2 | 128–319 | 155–319 |
| QB1 | passing TDs | 1,063 | 1.47 / 1.17 | 1.51 / 1.10 | 0–3 | 0–3 |
| QB1 | rushing yards | 1,063 | 18.8 / 20.1 | 19.7 / 18.5 | 0–48 | 0–43 |
| RB1 | rush attempts | 1,087 | 14.4 / 6.1 | 14.5 / 6.3 | 6–22 | 7–23 |
| RB1 | rushing yards | 1,087 | 63.0 / 39.5 | 62.2 / 36.9 | 18–114 | 20–112 |
| RB1 | receptions | 1,087 | 2.50 / 1.92 | 2.71 / 2.11 | 0–5 | 0–6 |
| RB2 | rush attempts | 940 | 6.6 / 5.1 | 7.2 / 5.2 | 1–14 | 2–14 |
| WR1 | targets | 1,080 | 7.56 / 3.43 | 7.45 / 3.59 | 3–12 | 3–12 |
| WR1 | receptions | 1,080 | 4.86 / 2.61 | 4.82 / 2.67 | 2–8 | 2–8 |
| WR1 | receiving yards | 1,080 | 62.0 / 39.6 | 60.1 / 39.7 | 15–113 | 15–114 |
| WR2 | receiving yards | 1,049 | 44.1 / 35.2 | 43.5 / 34.3 | 5–93 | 5–90 |
| WR3 | receiving yards | 1,015 | 26.2 / 27.0 | 29.3 / 28.4 | 0–62 | 0–68 |
| TE1 | targets | 1,067 | 4.98 / 3.08 | 4.89 / 3.16 | 1–9 | 1–9 |
| TE1 | receiving yards | 1,067 | 37.6 / 29.3 | 36.1 / 29.8 | 5–78 | 4–77 |

Receivers and running backs match closely. **QB1 volume is too high and too narrow** (attempts 33.0 vs 31.0, sd 7.1 vs 9.1): the sim never pulls the starter (injury, benching, blowout), and the expected starter was wrong or didn't play in about 7% of games. WR3 is about 12% high. Full table: `data/processed/player_wf_report.json`.

### Correlations (not hard-coded; same 2024–25 games)
| Pair | n team-games | Simulated | Actual |
|---|---|---|---|
| QB pass yds ~ WR1 rec yds | 1,011 | 0.450 | 0.448 |
| QB pass yds ~ WR2 rec yds | 982 | 0.376 | 0.409 |
| QB pass yds ~ TE1 rec yds | 995 | 0.303 | 0.361 |
| WR1 rec yds ~ WR2 rec yds | 994 | −0.043 | +0.022 |
| QB pass TDs ~ WR1 rec TDs | 1,011 | 0.443 | 0.455 |
| RB1 rush att ~ team margin | 1,087 | 0.295 | 0.284 |
| QB pass att ~ team margin | 1,063 | −0.204 | −0.173 |
| QB pass yds ~ game total | 1,063 | 0.426 | 0.434 |
| RB1 rush yds ~ QB pass yds | 1,017 | −0.027 | −0.085 |

Simulated correlations are pooled over all kept simulations (100 per game, active only); actual is one observation per team-game. Teammate receivers are slightly too negatively correlated (the Dirichlet makes them compete for a fixed pie), and QB–WR2/TE links are a bit weak.

### Parlay pricing
- **P(parlay) = K/N**: simulations where every leg won ÷ simulations. Reported with a 95% Wilson (Monte Carlo) interval. The product of leg win rates is shown only as an independence benchmark.
- Pairwise correlation of leg wins, plus per-leg win / push / void rates.
- **Settlement** (`--settlement`): `reduce` (default; pushed/void legs are dropped and the parlay pays the product of the remaining legs' offered odds, which needs `--leg-odds`; if all legs push/void the stake is refunded), `void_parlay` (any push/void refunds), `push_loses`.
- **EV/ROI** from the per-simulation payouts, using the odds *you* enter. No odds are invented: without `--leg-odds/--parlay-odds`, EV is not shown.
- **Model uncertainty**: 20 bootstrap refits (drive model resampled by game; each player's usage history resampled by game) × 2,000 sims, rerunning the parlay each time.

Example (BAL @ KC, 2024 week 1, models fit before 2024-09-05; no odds entered):
Mahomes over 249.5 pass yds 0.459, Rice over 55.5 rec yds 0.381 (void 0.039), KC ML 0.553 (push 0.005), over 46.5 0.525. Joint K/N = 1938/20000 = **0.0969** (95% MC CI 0.0929–0.1011). Independence benchmark 0.0507. Fair odds +932. Model uncertainty: bootstrap sd 0.0102 (0.0078 after removing Monte Carlo noise), 5–95% range 0.081–0.111.

## Simulation modes: independent vs market_anchored
| | `independent` (default) | `market_anchored` |
|---|---|---|
| Game view (mean margin, mean total) | the model's own | the market's spread and total |
| Joint structure (correlations, player splits, pace, game script) | simulator | simulator (unchanged) |
| Use for | evaluating the model; game-level backtests; games with no line | pricing props and parlays when you trust the market's game view more than the model's |

**Why:** the team model is clearly worse than the closing market (Brier 0.2208 vs 0.2131), but the simulator's joint structure matches reality well. Anchored mode combines the market's view of the game with the simulator's view of how legs move together.

**Spread sign convention (verified):** nflverse `schedules.spread_line` is the expected **home** margin, **positive = home favored**. It is the negative of the home team's sportsbook spread (home −3.5 ⇔ `spread_line` +3.5). Evidence on 2,877 games with lines:
- Its sign matches the moneyline favorite in 98.1% of games. All 56 mismatches have |spread| ≤ 1.5 and near-even moneylines.
- Home teams with `spread_line` ≥ 7 won by +10.6 on average; with ≤ −7, they lost by 9.6.
- The regression slope of actual home margin on `spread_line` is 1.03.

The CLI's `--spread` uses **sportsbook convention for the home team** (negative = home favored) and prints the conversion back.

**Lines:** backtests use the nflverse closing `spread_line` / `total_line`. For a single game, `--spread` / `--total` override nflverse. Otherwise the nflverse line is used and the output prints its source and as-of time. nflverse publishes no line timestamp, so the as-of shown is "closing line" for played games, or "latest line published by nflverse" plus the schedule download time for unplayed games. No line is ever invented. If the spread or total is missing, the game runs in independent mode and the output says "FELL BACK to independent".

**How anchoring works:** one additive log-odds shift per offense on its scoring outcomes (TD and FG) in the drive-outcome model, i.e. that offense's odds of a scoring drive × e^shift. A damped Newton/Broyden solver finds the two shifts so that the simulated mean home margin and mean total are each within **±0.25** of the line. It uses the same seed and N as the final run, and the final run *is* the last solver evaluation, so the tolerance holds for the reported simulation itself. Player shares, play templates, game-state effects, pace, field position and PAT behavior are not touched. The output reports each team's shift and the implied change in expected points per neutral drive, i.e. how far the market disagrees with the model.

Example (DAL @ PHI, 2026-10-26, not yet played, nflverse line PHI +4.5 / 43.5): the unanchored model says PHI −1.7 / 46.6. The anchor shifts PHI's offense by −0.319 log-odds (−0.39 expected points per drive) and DAL's by −0.041 (−0.05). Result: PHI −4.61 / 43.51, converged in 4 solver runs.

**Two fixes made while validating this mode:**
1. The first version shifted each team's `off_epa` input. 69 of 1,530 backtest games missed the tolerance (errors up to 7.6 points): the needed shifts were up to ~10 standard deviations outside the training range, and the logistic response saturated. Replaced with the scoring log-odds shift.
2. At 1,000 simulations, 26 of 570 games still missed (max 0.46). Changing the model changed how many random draws each drive consumed, so all later draws shifted and the solver was chasing Monte Carlo noise. The simulator now gives every simulation a fixed set of uniforms per drive step (common random numbers). This changes only which random numbers are used, not the model (see the M1 note above).

### Validation (walk-forward, same seasons and seeds)
**Anchoring check:**
- 2021 → 2026 week 7 (4,000 sims): all **1,530** games within ±0.25, no fallbacks. Max |margin error| 0.250, max |total error| 0.250, 4.6 solver runs on average.
- 2024–25 player run (1,000 sims): all **570** games within ±0.25 (max 0.249 / 0.248).
- Mean |market − model| gap: 2.16 points of margin and 2.67 points of total. The market total is on average 1.80 points *lower* than the model's.
- |Scoring log-odds shift|: mean 0.18, 95th percentile 0.45, max 1.22.

**Game-level metrics are not evidence of skill in anchored mode.** Anchored mode is fitted to the closing line, so matching the market is circular. They are shown only to confirm anchoring works: anchored Brier 0.2129, margin MAE 9.736, total MAE 10.328 vs closing market 0.2131 / 9.736 / 10.328. **Anchored-mode results cannot be used to claim the model beats (or matches) the market.**

**Known issues (simulated vs actual, same 1,530 games)** *(measured before the cleanup round; current numbers in "Cleanup round")*:

| | Actual | Independent | Anchored |
|---|---|---|---|
| Total bias (actual − simulated mean) | | −1.25 | +0.55 (closing market itself: +0.55) |
| Points per team-game | 22.6 | 23.2 | 22.3 |
| Total sd / margin sd | 13.7 / 14.1 | 12.8 / 13.9 | 13.0 / 14.1 |
| Final margin exactly 3 | 15.6% | 12.1% | 11.9% |
| Final margin exactly 7 | 8.0% | 7.2% | 7.2% |
| One-score games (≤ 7) | 49.8% | 45.4% | 44.6% |
| Overtime | 6.3% | 3.9% | 3.8% |

Anchoring fixes the total bias: it inherits the market's +0.55. Anchoring does **not** fix the close-game problems (too few 3-point margins and overtimes). Those come from the simulator's end-of-game behavior, not from team strength.

**Players, 2024–25 (1,000 sims/game), independent → anchored** *(before the cleanup round)*. Means, sds and p10–p90 move very little: by at most 3.4 yards for QB1 passing yards (234.7 → 231.4) and 1.1 yards for WR1 receiving yards (60.1 → 59.0). Every sd and percentile is within 1 unit / 4 yards. Full table: `python scripts/compare_modes.py`.

| Correlation | Actual | Independent | Anchored | Change |
|---|---|---|---|---|
| QB pass yds ~ WR1 rec yds | 0.448 | 0.450 | 0.454 | +0.004 |
| QB pass yds ~ WR2 rec yds | 0.409 | 0.376 | 0.382 | +0.005 |
| QB pass yds ~ TE1 rec yds | 0.361 | 0.303 | 0.308 | +0.004 |
| WR1 ~ WR2 rec yds | 0.022 | −0.043 | −0.034 | +0.010 |
| QB pass TDs ~ WR1 rec TDs | 0.455 | 0.443 | 0.448 | +0.004 |
| RB1 rush att ~ team margin | 0.284 | 0.295 | 0.297 | +0.002 |
| QB pass att ~ team margin | −0.173 | −0.204 | −0.212 | −0.008 |
| QB pass yds ~ game total | 0.434 | 0.426 | 0.422 | −0.004 |
| RB1 rush yds ~ QB pass yds | −0.085 | −0.027 | −0.030 | −0.003 |

No correlation moved by more than 0.03 (largest: 0.010).

## Cleanup round (before Milestone 3)
### 1. Drive-template leakage: none found
Templates, the pace table and the field-position table only ever see drives from games strictly before the cutoff:
- `team_model.py` `fit_team_model`: `d = drives[drives["game_date"] < cutoff]`. `PaceTable.fit(dpace)` and `FieldPosTable.fit(dpace)` use a subset of `d`; PAT rates use `pat["game_date"] < cutoff`.
- A template `drive_uid` can only come from `PaceTable.uid`. `players.Templates` holds every drive but is only indexed by those uids.
- The cutoff is the week's first game day in backtests and the game day for a single game.
- Regression test `test_templates_and_tables_only_from_past`: for 3 games, every table uid predates the cutoff, the game's own drives are absent, every template drawn in 2,000 simulations predates the cutoff, and the field-position table equals one rebuilt from strictly earlier drives.

No fix was needed, so the Milestone 2 validation was not re-run for this item.

### 2. QB early exits
**Estimated on 2018–2020** (`scripts/qb_exits.py`). The starter is the QB of the team's first dropback/kneel. A permanent exit is another QB taking a dropback/kneel after the starter's last one. There were 217 exits in 1,606 team-games (13.5%):
- **Blowout** (|margin| ≥ 17 in Q4 or ≥ 24 in the 2nd half at the exit): 124 (7.7%). 81 of them were by the *leading* team; 114 happened in Q4, 48 in the last 2:00.
- **Injury/benching** (no blowout): 93 (5.8%). 12 in Q1, 14 in Q2, 26 in Q3, 41 in Q4.
- Check only (not used for fitting): in 2021–25, 16.3% of team-games had an exit (10.0% blowout, 6.3% injury/benching). The 2018–20 rate is lower than the test window's.

**Simulation.** Before each drive, the starter can leave permanently with a per-drive hazard by state at the start of the drive:
- non-blowout states, by quarter: 0.33%, 0.26%, 0.76%, 1.03% for Q1–Q4;
- blowout buckets, split by leading/trailing. For example, leading by 24+ in the last 5:00: 51%; trailing: 15%.

After an exit:
- The backup (next healthy QB on the pregame depth chart; found for 98% of team-games) takes every remaining dropback, scramble, kneel and the starter's designed runs.
- The drive model uses the backup's rating for the rest of the game (`qb_epa` = backup's rating, `qb_delta` shifted by backup − starter).
- **After a blowout exit only**, other starters' usage drops, using role ratios measured on 2018–20 (`scripts/blowout_starters.py`; 450 targets / 329 carries after exits). For example, RB1 carry share is ×0.21 and WR1 target share ×0.60; depth players rise. Injury exits showed no such change (main-role ratios 0.84–1.16), so they keep shares.

**Grading.** The starter's props are graded on what he did before leaving; they are never voided by an exit. The backup's stats count only for the backup, who is active in every simulation where the starter left. Exits are only simulated when the depth chart names a backup. Simulated exit rate 2021+: 12.1% of team-games (actual 16.3%).

**Results, 2024–25 player walk-forward (1,000 sims/game, same seeds). Before = no exits; after = final exit model with the old late-game states, isolating this item:**

| QB1 (n = 1,063) | Actual | Indep. before | Indep. after | Anchored before | Anchored after |
|---|---|---|---|---|---|
| Pass attempts mean / sd | 31.0 / 9.1 | 33.0 / 7.1 | 31.9 / 8.1 | 33.0 / 7.1 | 31.9 / 8.1 |
| Pass attempts p10–p90 | 20–42 | 24–42 | 23–42 | 24–42 | 23–42 |
| Completions mean / sd | 20.2 / 6.3 | 21.6 / 5.0 | 20.9 / 5.6 | 21.5 / 5.0 | 20.8 / 5.7 |
| Completions p10–p90 | 13–28 | 15–28 | 14–28 | 15–28 | 14–28 |
| Passing yards mean / sd | 222.5 / 77.8 | 234.7 / 64.2 | 227.4 / 70.1 | 231.4 / 64.2 | 224.2 / 70.0 |
| Passing yards p10–p90 | 128–319 | 155–319 | 142–316 | 151–316 | 139–313 |

| Correlation (n team-games) | Actual | Indep. before | Indep. after | Anchored before | Anchored after |
|---|---|---|---|---|---|
| QB pass yds ~ WR1 rec yds (1,011) | 0.448 | 0.450 | 0.405 | 0.454 | 0.412 |
| QB pass yds ~ WR2 rec yds (982) | 0.409 | 0.376 | 0.336 | 0.382 | 0.346 |
| QB pass yds ~ TE1 rec yds (995) | 0.361 | 0.303 | 0.275 | 0.308 | 0.275 |
| WR1 ~ WR2 rec yds (994) | 0.022 | −0.043 | −0.040 | −0.034 | −0.036 |
| QB pass TDs ~ WR1 rec TDs (1,011) | 0.455 | 0.443 | 0.436 | 0.448 | 0.435 |
| RB1 rush att ~ team margin (1,087) | 0.284 | 0.295 | 0.283 | 0.297 | 0.282 |
| QB pass att ~ team margin (1,063) | −0.173 | −0.204 | −0.135 | −0.212 | −0.140 |
| QB pass yds ~ game total (1,063) | 0.434 | 0.426 | 0.385 | 0.422 | 0.387 |
| RB1 rush yds ~ QB pass yds (1,017) | −0.085 | −0.027 | −0.018 | −0.030 | −0.014 |

**QB1 marginals got much closer to actual. QB correlations got worse.** QB ~ WR1 / WR2 / TE1 / game total fell by 0.03–0.05 below actual, and QB attempts ~ margin went from too negative to not negative enough.

What I tried:
- The first exit model (symmetric blowout hazard, flat injury hazard) was worse: QB ~ WR1 0.383, attempts ~ margin −0.101. Splitting blowout hazards by leading/trailing and injury hazards by quarter (both measured on 2018–20) gave 0.403 / −0.135.
- The blowout starter-usage change added only +0.002 to QB ~ WR1.
- Hypothesis "passing efficiency collapses after an injury exit and the sim misses it" is **not** supported. In 2018–20, team passing yards per dropback went *up* after injury exits (5.61 → 6.06), and the dropback share went 59.5% → 67.7%.

**Split by exit vs no-exit games** (`scripts/exit_corr_split.py`, 2024–25, independent mode; actual n = team-games, simulated n = kept simulations):

| Correlation | Actual, no exit | Sim, no exit | Actual, exit | Sim, exit |
|---|---|---|---|---|
| QB pass yds ~ WR1 rec yds | 0.482 (823) | 0.445 (84,423) | 0.357 (188) | 0.303 (11,685) |
| QB pass yds ~ WR2 rec yds | 0.436 (801) | 0.377 (78,987) | 0.300 (181) | 0.241 (10,944) |
| QB pass yds ~ TE1 rec yds | 0.376 (811) | 0.306 (82,443) | 0.343 (184) | 0.200 (11,310) |
| QB pass yds ~ game total | 0.486 (864) | 0.434 (93,429) | 0.383 (199) | 0.260 (12,871) |

**Finding: the gap is spread across all games, not concentrated in exit games.** In no-exit games the sim is 0.04–0.07 too low on every pair. In exit games it is 0.05–0.06 too low for WR1/WR2 and 0.12–0.14 too low for TE1 and the game total. Before exits were modeled, the all-games QB ~ WR1 match (0.450 vs 0.448) came from two errors cancelling: the sim was already too low in normal games, and leaving out exits (which lower the actual all-games correlation) hid that. Exits are also rarer in the sim (12.2% of team-games vs 18.6% actual in this sample). The cause of the normal-game gap was not investigated further. For SGPs that pair a QB with his receivers, joint probabilities are probably somewhat understated. Note: the leading/trailing and by-quarter refinements were motivated by the 2024–25 results, but their parameters are estimated on 2018–20 only.

Game-level effect (2021+, independent): Brier 0.2208 → 0.2214, log loss 0.6319 → 0.6333, total MAE 10.734 → 10.716, total bias −1.25 → −1.14. The Brier/log-loss changes are small but in the worse direction.

### 3. Close-game endings
**Diagnosis on 2018–2020** (`scripts/close_games.py`: 803 actual games vs the same games simulated, 200 sims each). Actual 3-point finals (n = 111): 50.5% were decided by a FG kicked while tied (17% in the last 2:00, 18% in OT). OT games were tied by a FG from down 1–3 (45%) or a TD from down 4–8 (53%). The old model's late states pooled "trailing by 1–8", so it could not tell "needs a FG" from "needs a TD". Last 10 minutes of regulation, FG share of drives:

| Situation | Time left | Actual (n drives) | Old sim | New sim |
|---|---|---|---|---|
| Down 1–3 | ≤ 2:00 | 23.5% (102) | 9.2% | 19.1% |
| Down 1–3 | 2–5 min | 20.5% (78) | 12.6% | 25.8% |
| Down 4–8 | ≤ 2:00 | 0.0% (114) | 8.2% | 0.0% |
| Down 4–8 | 2–5 min | 0.7% (147) | 11.9% | 0.0% |
| Tied | ≤ 2:00 | 27.4% (73) | 23.5% | 22.7% |

**Fix.** Late-game states (Q4 and OT) are now score situation (down 9+, 4–8, 1–3, tied, up 1–3, 4–8, 9+) × time left (> 5:00, 2:00–5:00, ≤ 2:00). They are used by both the drive-outcome model and the pace/template table. The model's coefficients are estimated from play-by-play as before. Small pace cells fall back to the old coarse state before pooling across states.

**Keep test on 2018–2020 (803 games, 4,000 sims, final QB-exit model):** margin = 3 11.8% → 12.4% (actual 13.8%); one-score 43.9% → 44.5% (47.8%); OT 3.5% → 5.3% (4.7%: overshoots, but closer); ties 0.29% → 0.52% (0.50%). Brier 0.2157 → 0.2156, log loss 0.6212 → 0.6211, total MAE 10.900 → 10.902: no measurable change. **Kept.** Side effect: total bias +0.69 → +1.04 (simulated totals ~0.3 points lower).

**Known-issues table, 2021 → 2026 week 7 (1,530 games, 4,000 sims; actual first):**

| | Actual | Indep. start of round | Indep. after QB exits | Indep. final | Anchored start | Anchored after exits | Anchored final |
|---|---|---|---|---|---|---|---|
| Total bias (actual − sim) | | −1.25 | −1.14 | −0.84 | +0.55 | +0.55 | +0.55 |
| Margin = 3 | 15.6% | 12.1% | 12.2% | 13.1% | 11.9% | 11.9% | 12.7% |
| One-score (≤ 7) | 49.8% | 45.4% | 45.4% | 45.5% | 44.6% | 44.6% | 44.7% |
| Overtime | 6.3% | 3.9% | 3.8% | 6.0% | 3.8% | 3.8% | 5.8% |
| Ties | 0.33% | 0.35% | 0.36% | 0.65% | 0.38% | 0.39% | 0.68% |
| Brier | | 0.2208 | 0.2214 | 0.2214 | (0.2129)* | (0.2130)* | (0.2130)* |
| Log loss | | 0.6319 | 0.6333 | 0.6332 | (0.6132)* | (0.6137)* | (0.6137)* |
| Total MAE | | 10.734 | 10.716 | 10.678 | (10.328)* | (10.329)* | (10.333)* |

\* Anchored game metrics are a check that anchoring works, not skill (circular). Anchoring check on the final code: all 1,530 games within ±0.25 (max 0.249 / 0.250), and all 570 player-run games (max 0.250 / 0.249).

What improved: OT is now close to actual (6.0% vs 6.3%), 3-point finals moved partway (13.1% vs 15.6%), and total MAE improved. What did **not** improve: one-score share barely moved (45.5% vs 49.8%), and ties are now about double actual (0.65% vs 0.33%). The remaining shortfall: too few close games reach the final 2:00. The sim has 61 "down 1–3, ≤ 2:00" drives per 803 games vs 102 actual, and 34 tied ones vs 73. That's a clock/possession-count issue, not drive-outcome choice.

## Uncertainty
Two kinds, always reported separately:
- **Monte Carlo error** (finite simulations): Wilson CI / standard error. It shrinks as N grows.
- **Model error**: bootstrap refits for parlays. For game predictions, the gap to the closing market above is model error. **More simulations do not reduce model error.**

## Data gaps and fallbacks
- **Weather**: nflverse has actual game-time conditions, not forecasts. It is never used; every feature row is marked `weather = unavailable`.
- **Starting QB**: the schedule's QB columns are postgame facts and are used only for scoring the guess. The pregame QB is the depth-chart QB1 (weekly charts ≤2024; the latest timestamped snapshot before game day for 2025+). If that QB's final injury status is *Out*, the next QB on the chart is used. If there's no depth chart, the previous game's main passer is used. It matched the actual starter 89–96% of the time, depending on season.
- **Injury timing**: the final weekly report status is assumed to be known before kickoff. Out and Doubtful → inactive; Questionable → active (with the player's usual P(active)).
- **Game-day inactives** (90 minutes before kickoff) are not used; availability uncertainty is modeled with P(active) instead.
- **Prop lines and SGP prices**: not in nflverse. Nothing here is a real prop price; EV needs odds entered by the user. No prop ROI backtests.
- **Return TDs** are not credited to any player (anytime-TD probability is slightly low for returners).
- **Market lines**: in `independent` mode, closing lines are used only as a benchmark and never as features. In `market_anchored` mode, the spread and total are the anchor: the simulation is shifted to match them (see "Simulation modes"). They are still never inputs to the team, drive or player models.
- **Kickoff rule changes (2024, 2025)**: field position after kicks comes from the most recent 1,200 kicks, so the model takes part of a season to adapt after a rule change.
- Schemes, coaching changes and pressure/charting data are not used.

## Milestone 4 backlog: QB availability and backup quality (not started)
1. **Split the offense rating into QB and non-QB parts.** Rate the rest of the offense separately, then add the expected starter's rating. Currently `off_epa` still contains the previous QB's plays and is only corrected by `qb_delta`.
2. **Data-driven prior for QBs with little history.** The prior is currently **hand-set**: league-average EPA/dropback minus 0.10 (`RatingParams.qb_prior_offset`). Estimate it instead from how comparable QBs played in their first starts, using draft round, years in the league and depth-chart rank (nflverse draft picks and rosters).
3. **Tune the QB rating's prior strength and recency** (currently 250 dropbacks and a 12-game half-life, also not tuned). Tune on the 2018–2020 validation window only.
4. **Probability that the listed starter plays.** Use Questionable/Doubtful status and practice participation instead of the current Out-only rule.

Per the spec, keep each change only if the walk-forward shows it helps. Report QB-change games separately with sample sizes, since they are few (about 300–800 games since 2021, depending on definition) and the results are noisy.

## Backlog: end-of-game clock, possessions and timeouts (not started)
- Too few close games reach the last 2:00. Per 803 games (2018–20) the sim has 61 "down 1–3, ≤ 2:00" drives vs 102 actual, and 34 tied ones vs 73. Late close games end with too few possessions.
- The one-score-game share stays low (45.5% vs 49.8%, 2021+) even after the late-game score-situation states.
- The tie rate doubled with those states (0.65% vs 0.33% actual, 2021+).
- Timeouts are not modeled at all: no clock stoppages, no saving time for a final drive, no icing. Drive durations are resampled from history only by outcome, game state and field position.
Any change here follows the usual rule: tune on 2018–2020 only, and keep it only if margin = 3, one-score share, OT and ties move toward actual without measurably worse Brier, log loss or total error.

## Known limitations
- Totals run about 0.8 points high (independent mode). The model lags shifts in league scoring.
- Close games still reach the final two minutes too rarely, so 3-point finals (13.1% vs 15.6%) and one-score games (45.5% vs 49.8%) are low. Ties are about double actual (0.65% vs 0.33%).
- The number of drives isn't drawn from its own distribution. It comes out of the clock (sampled drive durations).
- Drive duration is resampled from empirical drives with the same outcome and state, not from team-specific pace.
- Defensive and special-teams TDs come only from the drive-outcome classes. There's no separate return model.
- Play templates are league-wide, not team-specific: a team's completion rate, sack rate and scramble rate follow the league mix for its drive outcomes. QB accuracy shows up only through drive outcomes, so QB completion props are less personalized than yardage/usage props.
- QB early exits are simulated (switchable with `--qb-exits`), but QB-to-receiver and QB-to-total correlations are 0.04–0.07 below actual even in games without an exit (cause not investigated). Temporary QB absences (starter returns) and mid-drive exits are not modeled. The simulated exit rate (12.1%) is below the 2021–25 actual (16.3%), because it was estimated on 2018–20.
- Player-layer settings (half-life 6, shrinkage strengths, P(active) window) were set by hand, not tuned. Only the share dispersion α was estimated (on 2023).
- Bootstrap model uncertainty does not refit the team EPA ratings or the empirical pace/field-position tables.
