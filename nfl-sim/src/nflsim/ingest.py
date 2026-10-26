"""Download and cache nflverse data, one parquet file per dataset per season.

Layout: data/raw/<dataset>/<season>.parquet

Re-running only downloads seasons that are missing on disk, plus the current
season (which is still changing). Team abbreviations are normalized so that
relocated franchises share one key (OAK->LV, SD->LAC, STL->LA). Players are
keyed by nflverse `gsis_id` (column `player_id` in pbp/player stats).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nfl
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MANIFEST = RAW / "manifest.json"

FIRST_SEASON = 2016

# Relocations and alternate spellings used across nflverse / PFR-derived sources.
TEAM_MAP = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
    "LAR": "LA",
    "JAC": "JAX",
}

# Columns holding team abbreviations, per dataset.
TEAM_COLS = {
    "schedules": ["home_team", "away_team"],
    "pbp": ["posteam", "defteam", "home_team", "away_team", "side_of_field",
            "timeout_team", "td_team", "return_team", "penalty_team"],
    "player_stats": ["team", "recent_team", "opponent_team"],
    "rosters": ["team"],
    "snap_counts": ["team", "opponent"],
    "depth_charts": ["team", "club_code"],
    "injuries": ["team"],
}

LOADERS = {
    "schedules": None,  # loaded once for all seasons, then split
    "pbp": nfl.load_pbp,
    "player_stats": nfl.load_player_stats,
    "rosters": nfl.load_rosters,
    "snap_counts": nfl.load_snap_counts,
    "depth_charts": nfl.load_depth_charts,
    "injuries": nfl.load_injuries,
}


def normalize_teams(df: pl.DataFrame, dataset: str) -> pl.DataFrame:
    cols = [c for c in TEAM_COLS.get(dataset, []) if c in df.columns]
    if not cols:
        return df
    return df.with_columns(
        [pl.col(c).cast(pl.Utf8).replace(TEAM_MAP).alias(c) for c in cols]
    )


def current_season() -> int:
    return int(nfl.get_current_season())


def _load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def _save_manifest(m: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2, sort_keys=True))


def path_for(dataset: str, season: int) -> Path:
    return RAW / dataset / f"{season}.parquet"


def ingest(seasons: list[int] | None = None, datasets: list[str] | None = None,
           force: bool = False, verbose: bool = True) -> dict:
    """Download missing / current seasons. Returns the manifest."""
    cur = current_season()
    seasons = seasons or list(range(FIRST_SEASON, cur + 1))
    datasets = datasets or list(LOADERS)
    manifest = _load_manifest()

    def need(ds: str, s: int) -> bool:
        return force or s == cur or not path_for(ds, s).exists()

    for ds in datasets:
        todo = [s for s in seasons if need(ds, s)]
        if not todo:
            if verbose:
                print(f"[{ds}] cached, nothing to do")
            continue
        (RAW / ds).mkdir(parents=True, exist_ok=True)

        if ds == "schedules":
            sched = normalize_teams(nfl.load_schedules(True), ds)
            for s in todo:
                part = sched.filter(pl.col("season") == s)
                part.write_parquet(path_for(ds, s))
                _record(manifest, ds, s, part.height, None)
                if verbose:
                    print(f"[{ds}] {s}: {part.height} rows")
            continue

        for s in todo:
            try:
                df = LOADERS[ds](s)
                df = normalize_teams(df, ds)
                df.write_parquet(path_for(ds, s))
                _record(manifest, ds, s, df.height, None)
                if verbose:
                    print(f"[{ds}] {s}: {df.height} rows")
            except Exception as e:  # dataset not published for this season yet
                _record(manifest, ds, s, 0, f"{type(e).__name__}: {e}")
                if verbose:
                    print(f"[{ds}] {s}: MISSING ({type(e).__name__}: {str(e)[:120]})")
        _save_manifest(manifest)
    _save_manifest(manifest)
    return manifest


def _record(m: dict, ds: str, s: int, rows: int, err: str | None) -> None:
    m.setdefault(ds, {})[str(s)] = {
        "rows": rows,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "error": err,
    }


def load(dataset: str, seasons: list[int] | None = None) -> pl.DataFrame:
    """Read cached seasons for a dataset (missing seasons are skipped)."""
    d = RAW / dataset
    files = sorted(d.glob("*.parquet"))
    if seasons is not None:
        files = [f for f in files if int(f.stem) in set(seasons)]
    if not files:
        raise FileNotFoundError(f"No cached data for {dataset}; run ingest first")
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Download/cache nflverse data")
    p.add_argument("--seasons", type=int, nargs="*")
    p.add_argument("--datasets", nargs="*", choices=list(LOADERS))
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    ingest(a.seasons, a.datasets, a.force)


if __name__ == "__main__":
    main(sys.argv[1:])
