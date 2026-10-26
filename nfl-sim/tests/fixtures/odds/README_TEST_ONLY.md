# TEST-ONLY fixtures — not real odds

These JSON files are small hand-written examples in the response shapes documented
by The Odds API v4 (event objects with `bookmakers -> markets -> outcomes`;
historical responses wrapped in `timestamp` / `previous_timestamp` / `next_timestamp` / `data`).
**All prices and lines are made up.** They exist only
so the tests can exercise parsing, mapping, de-vig, consensus and budget logic
without calling the real API. Nothing outside `tests/` may read them.
