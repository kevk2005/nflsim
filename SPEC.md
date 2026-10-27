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
- Work through the milestones **in the order listed** below, and skip anything marked **DONE**.
- **Stop and wait for the user** after each milestone, and after any step that spends Odds API credits.

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
    pipeline.py      # loads inputs once; simulate_game() in either mode
    anchor.py        # market-anchored mode (Milestone 2.5)
    odds_feed.py     # The Odds API feed + credit budget (Milestone 2.6)
  scripts/           # diagnostics and estimation scripts (2018-2020 tuning window, etc.)
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
**Status: DONE**

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
**Status: DONE**

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

## Milestone 2.5: Market-anchored mode
**Status: DONE**

- Two simulation modes everywhere (pipeline, parlay CLI, backtests): `independent` (the model's own game view; default) and `market_anchored`.
- Anchored mode matches the market's spread and total by adding one log-odds shift per offense on its scoring outcomes (TD and FG) in the drive-outcome model. The two shifts are solved numerically (Newton/Broyden, common random numbers) so the simulated mean margin and mean total are each within **±0.25** of the line. Nothing else changes, so correlations still come from the shared simulation.
- Verified convention: nflverse `spread_line` = expected home margin (positive = home favored). No line → falls back to independent mode and says so.
- **Anchored game-level metrics are only a check that anchoring works, not evidence of skill** (they are fitted to the market).

## Cleanup round
**Status: DONE**

- **Template leakage:** confirmed clean. Play templates, pace and field-position tables come only from drives before the cutoff (regression test added).
- **QB early exits:** the starter can leave a game (blowout pulls and injury/benching, hazards estimated on 2018–2020), the depth-chart backup takes over with his own rating, and other starters' usage drops after blowout pulls. The starter's props grade on what he did before leaving. Switch: `--qb-exits on|off` (default on).
- **Late-game states:** score situation (down 9+, 4–8, 1–3, tied, up 1–3, 4–8, 9+) × time left (> 5:00, 2:00–5:00, ≤ 2:00) in the drive-outcome model and pace table; kept after a 2018–2020 test.
- **Open issues:**
  - QB–receiver (and QB–game-total) correlations are 0.04–0.07 below actual, in all games (with and without a QB exit).
  - The tie rate is about double actual (0.65% vs 0.33%).
  - Too few close games reach the last 2:00, so 3-point finals and one-score games stay low.

## Milestone 2.6: Odds feed and line history (approved addition)
**Status: DONE**
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

## Milestone 2.7: Correlation fix
**Status: DONE (shock built, not kept: off by default)**

- Step 1 supported, with a refinement. On 2018–2023, the missing game-to-game variation is in pass volume/pass rate (realized ÷ simulated variance: pass rate 1.79, attempts 1.40, yards per attempt only 1.08). About half of the within-game volume correlation is a persistent team pass tendency, the rest game-specific.
- Step 2: `--pass-shock on|off` (per-team, per-simulation pass-share tilt in template selection; size 0.4 estimated on 2018–2023).
- Step 3: QB–receiver and WR1–WR2 correlations move clearly toward actual (2024–25: QB~WR1 0.405 → 0.438 vs 0.448 actual). But RB1/WR1/TE1 spreads widen beyond actual, and 2021+ total MAE worsens by +0.053 (CI +0.030 to +0.076). Left off; details in README.

Hypothesis: real games have a shared per-game passing shock (one team's whole passing game runs hot or cold together), which the simulator lacks, so QB–receiver correlations come out 0.04–0.07 too low in all games.

1. **Test the hypothesis first, on 2018–2023 only.** Measure whether team passing efficiency/volume varies game to game more than the simulator produces, and whether that excess explains the QB–receiver correlation gap. Report the numbers.
2. **Build only if supported:** a shared per-game passing shock, behind a `--pass-shock on|off` switch.
3. **Keep only if** QB–receiver correlations move clearly toward actual **without** widening player stat distributions beyond actual and **without** hurting game metrics (Brier, log loss, score error). Otherwise leave it off and record the result.

## Milestone 3: Backtest and interface
**Status: DONE**

- Prop/SGP walk-forward, 2024 → 2026 wk 7 (676 games), 4 configs, pass shock off.
  - Kept: market_anchored, QB exits off. The margin is tiny, and exits off overstates QB volume.
  - Props at self-set lines are well calibrated (135,084 props; every bucket within ~1.2 points).
  - Synthetic SGPs: K/N beats the independence product (log loss 0.5245 vs 0.5388) but still runs about 1.5 points low (25,626 parlays, noisy).
- Real-line check (106 games, 2026 backfill): the market beats the model in every prop market (Brier 0.2597 vs 0.2469), and the model adds no detectable information beyond the market line. ROI on 13,103 bets is −2.6% (CI −6.6% to +1.8%): not a betting signal.
- App: market browser (default fanatics), parlay builder with K/N, CI, independence benchmark, fair/offered odds, ROI, correlations, MC vs model uncertainty and missing data. It warns on QB + own WR/TE, greys out what it can't price, grades and logs parlays, and has a backtest page. Alternate lines supported (`pull-props --alternates`, ≤ 8 extra credits/game) but not pulled.

**Build:**
1. `backtest.py`: retrain **once per week** using only data before that week. For each historical game, simulate and record predictions. Report: win probability calibration, score error, prop Brier score and log loss at self-set lines, calibration charts by market and probability bucket, results by season. For parlays, generate synthetic same-game parlays from historical games and check calibration of joint probabilities; report sample sizes and note that this check is noisy.
   - **Configurations:** props and synthetic parlays are evaluated in both modes (`independent` and `market_anchored`) and with each switch on and off (`--qb-exits`, `--pass-shock` if Milestone 2.7 keeps it). Keep whichever configuration calibrates best, and report all of them.
   - **Prop ROI:** uses the backfilled 2026 lines (the API's snapshot before kickoff) plus the saved live pre-kickoff lines, priced at those saved prices and settled against official results. No self-set lines, no invented prices. **Until enough games are collected, Milestone 3 evaluates calibration only** (win probability, scores, props at self-set lines, synthetic SGPs); any ROI figure must state how many games and props it covers. Same-game-parlay ROI is not possible: the feed has no real SGP prices.
2. `app/streamlit_app.py`: **market browser and parlay builder.**
   - Pick a game and a book (default **fanatics**). Markets come from the saved odds snapshots, grouped like a sportsbook: game lines first, then by player.
   - Legs are clickable, including alternate lines. (Alternate lines need the `_alternate` prop market keys pulled. They are billed per market, so plan their credit cost before pulling.)
   - Markets the simulator can't price (e.g. longest reception, first TD, quarters, kicker, defense) are shown greyed out and are never guessed. Only markets present in saved snapshots can be shown.
   - The user enters the book's SGP price, and can adjust the simulation count.
   - A run shows: each leg's probability, the joint probability with CI, the independence benchmark, fair odds, offered odds, expected ROI, pairwise correlations, Monte Carlo vs model uncertainty, and the missing data used.
   - After the game: grade the parlay against actual results, and keep a log of the user's parlays.

**Done when:**
- The backtest runs end to end and produces real calibration results.
- The app runs locally, browses a game's saved markets for a chosen book, evaluates a multi-leg parlay for an upcoming or historical game, and grades and logs parlays after the game.

## Milestone 4: Game report (tail scanner page)
**Status: DONE (built; keep test FAILED: one low-level calibration band)**

**Result** (2024 → 2026 wk 7; 676 games, 1,411,452 graded player-thresholds, 94,742 player-stat results):
- **Calibration:**
  - low **FAIL**: the 40–50% band is sim 44.9% vs actual 41.8%; every low band above 10% runs 1.7–3.1 points high;
  - mid PASS (max 2.0 points);
  - high PASS (max 1.8 points).
- **Percentiles, pooled: PASS.** 12.1% below p10, 52.7% below median, 9.6% above p90.
- **By stat:**
  - percentiles PASS for pass TDs, targets, receptions, rec yds, rec long and TDs;
  - FAIL for pass yds, pass att, completions, INTs, rush att, rush yds and rush long. QB attempts/completions are below the sim's p10 19–20% of the time: short real outings that the kept config (QB exits off) leaves out.
- **Trusted stat × level:** pass TDs low/high, targets mid/high, receptions high, rec yds low/high, rec long mid/high. Everything else is greyed in the app with its reason.
- **No tuning.**


**Earlier version (superseded the same day).** The first Milestone 4 flagged player-thresholds where the sim was above the player's baseline or the market. Its keep test FAILED on market edge: 133 sim-above-market spots, actual minus market −5.7 points (CI −11.9 to +1.0), ROI −31.9%. High-threshold calibration passed (largest bucket error 1.1 points). That keep test is retired. Market comparison is now an info column only.

**Goal:** one full report per upcoming game showing everything the simulator says happens, with the data behind each number. No probability cutoff.

**1. Game report**
- **Team / game level:**
  - win probability;
  - spread (home margin), total and each team's points: mean, median, 10th/25th/75th/90th percentiles;
  - drives, plays and pass rate per team;
  - P(team scores 20+ / 27+ / 34+).
  - Probabilities carry a Monte Carlo 95% range and a bootstrap (model) range.
- **Player level:** every player the sim tracks (both teams; active in at least 100 simulations).
  - Stats: pass yds, pass att, completions, pass TDs, INTs, rush att, rush yds, rush long, targets, receptions, rec yds, rec long, TDs. Each stat applies to its positions.
  - For each stat: mean, median, 10th/25th/75th/90th percentiles over the simulations where the player is active.
  - A player's stat is left out when it is zero in every active simulation (e.g. a backup QB's passing with QB exits off).
  - Hit % at every threshold on a finer grid (config `nfl-sim/config/tail_thresholds.toml`):
    - every 5 yards for rush/rec yards and longest plays;
    - every 25 for passing yards;
    - every 1 for counts.
  - Each threshold carries a level by height, fixed in the config before the backtest (e.g. rec/rush yards: low < 60, mid 60–95, high 100+).
  - Each hit % has a Monte Carlo 95% range (Wilson) and a bootstrap model range (± 1.96 × the model sd from 10 refits, with Monte Carlo noise removed).

**2. Data behind each number**
- **Role baseline:** hit rate of a typical player in the same pregame role (QB1, RB1, WR2, TE1, ...) over the previous two seasons, before the game.
- **Player history:** game log of his last 16 games played, plus his own hit rate shrunk toward the role baseline with a prior worth 8 games.
- **Usage inputs:** the model's actual inputs, compared with the role/position average, with the main drivers named.
  - Receivers and backs: target share, catch rate, yards-bin mix (share of catches of 20+ yards), TD-rate multiplier, carry share, share of carries of 10+ yards, P(active).
  - QB passing: comes from the team drive model, not from player-level rates.
- **Opponent context:** the player model does **not** use opponent defense by position, and that is not added. The opponent's overall defense (opponent-adjusted EPA and success rate allowed) enters the team drive model and is shown.
- **Lift** = sim % − role baseline %. The report sorts by lift, hit % and threshold height.
- **Market %:** info only, from saved prices.

**3. Correlations**
- Per game: the strongest positive and negative correlations across simulations between legs. Legs are team win, total, team points and the main players' stats.
- Same-player pairs and game-level identities are excluded from the top lists but kept in the export.

**4. Output**
- The Streamlit "Tail scanner" page becomes the game report: game picker, team section, expandable player sections, sorting and filters.
- Each game is exported to `nfl-sim/reports/<game_id>/` as CSV tables plus a readable `report.html`.

**Backtest:**
- 2024 → 2026 wk 7, weekly refits, no peeking.
- Pass shock off; the Milestone 3 kept config (market_anchored, QB exits off).
- 2,000 simulations per game, Monte Carlo only (no bootstrap refits in the backtest).
- Graded only when the player played. Actuals are official stats; longest plays come from play-by-play.

**Keep rule (written 2026-10-27, before the backtest is run):**
- **Calibration bands:** sim probability in [0, .1), [.1, .2), …, [.9, 1.0].
  - A band passes if |mean sim − actual rate| ≤ 0.03.
  - Bands with n < 200 are reported but not scored.
  - Each band reports a 95% CI on the actual rate from a bootstrap that resamples games.
- **Percentile check:** randomized PIT (P(X < actual) + U·P(X = actual)), so ties in counts are handled.
  - Share below the sim's 10th percentile (PIT < .1) must be within 3 points of 10%.
  - Share below the median (PIT < .5) must be within 3 points of 50%.
  - Share above the 90th percentile (PIT > .9) must be within 3 points of 10%.
  - Scored rows: the player is active in ≥ 50% of simulations, and the stat is nonzero in ≥ 50% of his active simulations (≥ 25% for TDs). Other rows are reported only.
- **Overall PASS needs both:**
  - (a) every scored band passes within each level (low, mid, high), pooling stats;
  - (b) the percentile check passes pooled over all stats.
- **Trust labels:** reported by stat type, by level, and by stat × level.
  - A stat × level is trusted when all its scored bands pass and its stat's percentile check passes.
  - "Not enough data" means no scored band.
  - The app greys out untrusted stat × level rows and gives the reason.
- **If it fails, it is reported as a failure. No tuning until it passes.**

## Milestone 5: More pregame context (in priority order)
**Status: NOT STARTED**

Add one at a time, and keep each only if the walk-forward test shows it helps:
1. QB availability and backup quality.
2. Other injuries to key players (WR1, RB1, offensive line, top pass rushers).
3. Travel distance and time zones.
4. Weather forecasts for upcoming games only (from a forecast API), clearly marked as unavailable in backtests.
5. Strength of schedule and regression toward underlying performance (records and streaks are not direct evidence of ability).

## Backlog (not scheduled)
- **End-of-game clock, possessions and timeouts:** too few close games reach the last 2:00, the one-score share is low (45.5% vs 49.8%), ties are about double actual (0.65% vs 0.33%), and timeouts are not modeled.
- **QB exit rate below actual:** simulated 12% of team-games vs 16–19% actual (the hazards were estimated on 2018–2020, when exits were rarer).
- **Team pass tendency / clock-neutral pass shock (from Milestone 2.7):** about half the within-game pass-volume correlation is a persistent team pass tendency (pregame-predictable; templates are league-wide). A shock that leaves run volume and drive clock unchanged might avoid 2.7's side effects. Each needs its own keep test.
- **Prop prices on real lines (from Milestone 3):** the model is overconfident against book lines and adds no information beyond them. Candidates: shrink player-level means toward the market line, or widen player distributions where the market disagrees. Each needs its own walk-forward test on saved lines.
- **Unmatched odds player names:** 13 names remain from the 2026 backfill (e.g. "Joshua Palmer"). Each needs a by-hand roster check, then an entry in the reviewed alias list (`odds_feed.REVIEWED_ALIASES`; "Hollywood Brown" → Marquise Brown was added 2026-10-26). No fuzzy matching.
- **Milestone 5 QB items (from the README backlog):** split the offense rating into QB and non-QB parts; a data-driven prior for QBs with little history (the current −0.10 EPA prior is hand-set); tune the QB rating's prior strength and recency on 2018–2020; use Questionable/Doubtful status and practice participation for the probability that the listed starter plays.

## Final report (after each milestone and at the end)
- How to install and run everything (data download, tests, backtest, app).
- What works and what has been validated, with the actual numbers.
- What data is missing and what fallbacks are used.
- Known limitations and modeling assumptions.
- Do not claim accuracy or profitability that the results don't show.