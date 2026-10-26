# Build: NFL Monte Carlo Game Simulator for Props and Parlays

## Goal
Build a Python system that simulates NFL games thousands of times from pregame information and uses those shared simulated games to estimate the probability that individual betting legs (props, spreads, totals) and whole parlays hit.

The key idea: every leg in a parlay is graded against the **same simulated game**, so correlation between legs comes out of the simulation itself. Never multiply leg probabilities to get a parlay probability.

## How to work
- Work **one milestone at a time**, in order. Do not start the next milestone until the current one meets its "Done when" criteria.
- At the end of each milestone, stop and report: what you built, what tests pass, what validation results you got (real numbers only), and what's missing.
- If a repository already exists, inspect it first and keep its working features and structure. If not, create the layout below.
- Write real, running code. No pseudocode, placeholders or mock data presented as real.
- Keep it simple first. A working simple version beats a half-built complex one.

## Hard rules (apply to every milestone)
1. **No future data.** Every prediction for a game may only use information available before that game. Every feature carries the date it became known. Never fill a missing pregame value with postgame data.
2. **No fabrication.** Never invent odds, injuries, weather, coaching changes, correlations, accuracy numbers or profits. If data is missing, mark it missing, use a documented fallback, and say so in the output.
3. **Shared game outcomes.** Player stats must be consistent with the simulated game: receptions ≤ targets, team completions = sum of player receptions, receiving yards come only from completed passes, touchdowns match scoring plays, team points match the scoring events.
4. **Parlay probability = K/N**, where N is the number of simulations and K is the number in which every leg won. Report a confidence interval for this.
5. **Two kinds of uncertainty.** Monte Carlo error (shrinks with more simulations) and model error (does not). Always report them separately and make clear that more simulations do not fix a bad model.
6. **Chronological validation only.** Never use random train/test splits.

## Tech stack
- Python 3.11+
- `nflreadpy` (or `nfl_data_py` if needed) for nflverse data
- `pandas` or `polars`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`
- Parquet files for local storage
- `streamlit` for the interface
- `pytest` for tests

## Project layout
```
nfl-sim/
  data/raw/          # downloaded nflverse files, by season
  data/processed/    # cleaned tables
  src/nflsim/
    ingest.py        # download + cache data (incremental by season)
    features.py      # pregame feature rows with as-of dates
    team_model.py    # team strength + scoring model
    simulator.py     # drive-level game simulator
    players.py       # splits team volume among players
    markets.py       # leg definitions + grading (over/under, push, void)
    parlay.py        # joint evaluation across simulations
    odds.py          # odds conversion, fair odds, EV
    backtest.py      # weekly walk-forward evaluation
  app/streamlit_app.py
  tests/
  README.md
```

## Known data gaps (do not try to fabricate these)
- **Player prop lines and same-game parlay prices:** not in nflverse. Prop calibration must use self-set lines (e.g. each player's projected median, or fixed lines). No prop ROI backtests except on real lines from the odds feed (Milestone 2.6): the one-time 2026 backfill plus our own live pre-kickoff pulls going forward.
- **Weather forecasts:** nflverse has game-time conditions, not forecasts. Do not use actual conditions as a pregame feature in backtests; mark weather as unavailable historically.
- **Schemes and coaching changes:** not available. Skip.
- **Pressure and advanced charting:** only recent seasons. Use only where present; mark missing otherwise.
- **Injury reports:** weekly reports exist, but exact timing is approximate. Treat the final weekly status as known before kickoff and document that assumption.
- **Game lines:** nflverse schedules include closing spread, total and moneylines. These may be used as a market comparison benchmark.

---

## Milestone 1: Data, team model, and game simulator

**Build:**
1. `ingest.py`: download schedules/results, play-by-play, weekly player stats, rosters, snap counts, depth charts and injuries for 2016 to present. Cache by season; re-running only downloads missing or current seasons. Normalize team abbreviations (relocations) and use nflverse player IDs (`gsis_id`) as the player key.
2. `features.py`: for each game, build team features using only prior games: offensive and defensive EPA/play and success rate, with recency weighting and shrinkage toward league average, adjusted for opponent strength. Add home field, rest days and the starting QB. Each feature row stores its as-of date.
3. `team_model.py`: start with a simple rating model (Elo or a regression on margin). Then fit models that give each team's expected points per drive and drive outcome probabilities (touchdown, field goal, punt, turnover, end of half) based on matchup.
4. `simulator.py`: **drive-level** simulator. Each simulated game alternates possessions, samples the number of drives, and samples each drive's outcome and play count from the matchup model. Track score and clock by quarter. Teams that are trailing late should become more aggressive (more passing, faster pace); teams leading late should run more and slow down. Estimate these effects from play-by-play.

**Done when:**
- A simulated game produces a valid score and drive log for any matchup.
- Simulated league-wide numbers match history within reasonable ranges: points per game, drives per game, home win rate, score distribution, how often favorites win.
- Weekly walk-forward test on 2021–present reports win probability Brier score and log loss and score error for: your model, a home-team baseline, and the closing market line. Report the real numbers, whatever they are.

## Milestone 2: Players, markets, parlays and odds

**Build:**
1. `players.py`: within each simulated drive, sample play types (pass/run) and outcomes, then assign each play to a player using their recent target share, carry share and snap share (shrunk toward positional averages). Pass plays: target → completion or not → yards → touchdown or not. Run plays: carrier → yards → touchdown or not. QB stats are the sum of his pass plays. Inactive or out players get zero; their share is redistributed to teammates.
2. `markets.py`: support passing yards, passing TDs, pass attempts, completions, rushing yards, rush attempts, receptions, receiving yards, anytime TD, plus game spread, total and moneyline. Grade each leg as win / loss / push / void (void if the player does not play).
3. `parlay.py`: for N simulations, grade every leg in every simulation. Output: each leg's hit rate, the joint hit rate (K/N) with confidence interval, the independence benchmark (product of leg rates, shown for comparison only), pairwise correlation between legs, and push/void handling.
4. `odds.py`: American ↔ decimal ↔ implied probability conversion, removing the vig from two-sided lines, fair odds from a probability, expected profit and ROI for given offered odds.
5. Settlement rules: default is that a void or pushed leg is removed and the parlay pays at the reduced odds. Make this configurable.
6. Model uncertainty: refit the team and player models on bootstrap resamples (start with 20) and rerun the parlay; report the spread of results as model uncertainty.

**Done when:**
- Consistency tests pass on thousands of simulated games (rule 3 above).
- Simulated player stat distributions (mean and spread) roughly match historical distributions for players with similar roles.
- Known correlations appear naturally without being hard-coded: e.g. a QB's passing yards and his top WR's receiving yards move together.
- Unit tests pass for odds conversion, push and void handling, payout math, aggregation and K/N calculation.

## Milestone 2.6: Odds feed and line history (approved addition)
Source: The Odds API (paid plan, 20,000 credits/month). Credits cost real money, so budget controls come first. **Live pulls for upcoming games, plus one approved, one-time historical backfill of the 2026 season to date** (completed games before 2026-10-26). After that backfill the historical endpoint is disabled again unless another one is explicitly approved. Every live pull is saved with its UTC timestamp, so the project builds its own line history going forward.

**Build:**
1. `odds_feed.py`: key from `ODDS_API_KEY` in `.env` (git-ignored; never printed or logged). Endpoints, NFL sport key, prop market names, bookmaker keys and credit costs are taken only from the official docs.
2. Budget controls: monthly caps per category (defaults: live game lines 2,000, live props 16,000, reserve 2,000 never spent automatically). Estimate every call's cost first; refuse if it would exceed its cap or drop remaining credits below the reserve. Log every call (time, endpoint, estimated and actual credits, remaining). `--dry-run` everywhere; a `credits` command shows usage by category this month.
3. Live game lines: moneyline, spreads, totals for upcoming NFL games, US books, American odds; raw JSON saved with UTC pull time; normalized to `odds_snapshots.parquet` (game × book × market × pull time); games mapped to nflverse `game_id` by teams and kickoff date (unmatched logged, never guessed); per-snapshot consensus (median spread/total, de-vigged moneyline).
4. Live player props for the `markets.py` markets, per book (player, market, line, over/under prices, pull time); players mapped to `gsis_id` by name, team and position (unmatched logged, never guessed; match rate reported); two-sided props de-vigged.
5. Optional `--bookmaker KEY` filter on every pull (docs: `bookmakers` takes priority over `regions`; up to 10 bookmakers = 1 region, so one book costs the same as the `us` region).
6. Integration: market-anchored line priority = user `--spread/--total` → latest pre-kickoff odds snapshot → nflverse → independent fallback, printing source, pull time and a warning if older than 6 hours. No snapshot pulled after a game's kickoff is ever used for that game. The parlay CLI can price legs from the latest snapshot (printing book and pull time); user-entered odds override.
7. One-time backfill (approved revision): 2026 completed games only; one snapshot per game, the closest available before kickoff (request `date` = kickoff − 1 min), game lines (one call per kickoff slot) plus the 9 supported prop markets (one call per game); 10 US books in one request including fanatics (docs: up to 10 bookmakers = 1 region); historical cap 14,000 for 2026-10 only, then 0; `--dry-run` first and `--confirm` only after approval; resumable and idempotent (never pays twice for a saved game); the API's snapshot time stored on every row; any call billed above its estimate stops the run.
8. CLI: `pull-lines`, `pull-props`, `latest [--game-id]`, `credits`, `backfill`, `check-books`, all with `--dry-run`.

**Done when:**
- Tests (on small saved example responses, test-only, no real API calls) cover parsing, game/player mapping, consensus, de-vig, budget caps, reserve floor, dry-run, the bookmaker filter, saved line history, refusal of historical endpoints without an active approval, the one-time backfill (10-book request, snapshot times, resume/idempotency, cap stop, empty-snapshot step-back, cost-overrun stop, approval consumed on completion), the stale-snapshot warning and the no-post-kickoff rule.
- One real live snapshot is pulled and reported (games, books, props, player match rate, credits used).

## Milestone 3: Backtest and interface

**Build:**
1. `backtest.py`: retrain **once per week** using only data before that week. For each historical game, simulate and record predictions. Report: win probability calibration, score error, prop Brier score and log loss at self-set lines, calibration charts by market and probability bucket, results by season. For parlays, generate synthetic same-game parlays from historical games and check calibration of joint probabilities; report sample sizes and note that this check is noisy.
   - **Prop ROI (updated after Milestone 2.6):** prop ROI backtesting uses the backfilled 2026 lines (the API's snapshot before kickoff) plus every line we save going forward from live pre-kickoff pulls, priced at those saved prices and settled against official results. No self-set lines, no invented prices. **Until enough games are collected, Milestone 3 evaluates calibration only** (win probability, scores, props at self-set lines, synthetic SGPs); any ROI figure must state how many games and props it covers. Same-game-parlay ROI is not possible: the feed has no real SGP prices.
2. `app/streamlit_app.py`: parlay builder. User picks a game, then adds legs (player, market, over/under, line), enters the sportsbook's offered odds, sets the simulation count, and runs. Show: each leg's probability, joint parlay probability with confidence interval, independence benchmark, fair odds, offered odds, expected ROI, pairwise correlations, Monte Carlo error vs model uncertainty, and a list of missing data used in this prediction.

**Done when:**
- The backtest runs end to end and produces real calibration results.
- The app runs locally and evaluates a multi-leg parlay for an upcoming or historical game.

## Milestone 4: More pregame context (in priority order)
Add one at a time, and keep each only if the walk-forward test shows it helps:
1. QB availability and backup quality.
2. Other injuries to key players (WR1, RB1, offensive line, top pass rushers).
3. Travel distance and time zones.
4. Weather forecasts for upcoming games only (from a forecast API), clearly marked as unavailable in backtests.
5. Strength of schedule and regression toward underlying performance (records and streaks are not direct evidence of ability).

## Final report (after each milestone and at the end)
- How to install and run everything (data download, tests, backtest, app).
- What works and what has been validated, with the actual numbers.
- What data is missing and what fallbacks are used.
- Known limitations and modeling assumptions.
- Do not claim accuracy or profitability that the results don't show.