"""Team strength and drive-outcome models.

1. Elo: a simple results-based rating (as-of, sequential).
2. DriveModel: multinomial logistic regression for drive outcome given the
   matchup (offense ratings vs defense ratings, QB, Elo), field position,
   clock and score state.
3. Empirical tables estimated from play-by-play:
   - PaceTable: (plays, seconds, pass share) per drive by outcome and game state.
     This is where "trailing late -> pass more / hurry" and "leading late ->
     run more / slow down" comes from.
   - FieldPosTable: next drive's starting yard line given the previous drive's
     outcome and where it started (kickoffs pooled from recent drives).
   - PATModel: go-for-two rate by score state, XP and 2-pt success rates.

`fit_team_model(cutoff)` fits everything using only data dated before `cutoff`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

OUTCOMES = ["TD", "FG", "MISSED_FG", "PUNT", "TURNOVER", "DOWNS", "END_HALF", "SAFETY", "OPP_TD"]
OUT_IDX = {o: i for i, o in enumerate(OUTCOMES)}

# Late game = 4th quarter (and overtime). The exact score situation matters late:
# down 1-3 needs a field goal, down 4-8 a touchdown (the 2018-20 diagnosis in
# scripts/close_games.py), so late states are score situation x time left.
LATE_SITS = ["down9", "down4_8", "down1_3", "tied", "up1_3", "up4_8", "up9"]
LATE_TIMES = ["q4", "5m", "2m"]            # >5:00 left, 2:00-5:00, <=2:00
STATES = ["normal", "h1_2m"] + [f"{s}_{t}" for t in LATE_TIMES for s in LATE_SITS]
STATE_IDX = {s: i for i, s in enumerate(STATES)}


def state_index(half: np.ndarray, half_secs: np.ndarray, diff: np.ndarray) -> np.ndarray:
    """Game-state bucket (vectorized). `diff` is offense score minus defense score.

    half: 1, 2, or 3 (overtime: late game, timed by the OT clock).
    """
    half = np.asarray(half)
    half_secs = np.asarray(half_secs, dtype=float)
    diff = np.asarray(diff, dtype=float)
    late = (half >= 2) & ((half_secs <= 900) | (half == 3))
    two = half_secs <= 120
    sit = np.select([diff <= -9, diff <= -4, diff <= -1, diff == 0, diff <= 3, diff <= 8],
                    [0, 1, 2, 3, 4, 5], 6)
    t = np.select([half_secs <= 120, half_secs <= 300], [2, 1], 0)
    out = np.zeros(half.shape, dtype=int)
    out = np.where((half == 1) & two, 1, out)
    out = np.where(late, 2 + t * len(LATE_SITS) + sit, out)
    if COARSE_LATE:
        out = coarse_state(out)
    return out


# Reproduce the pre-change late-game states (for before/after comparisons only):
#   NFLSIM_COARSE_LATE=1  -> trailing big / trailing 1-8 / tied / leading 1-8 /
#   leading big x (>2:00, <=2:00), mapped onto the first 12 state indices.
COARSE_LATE = os.environ.get("NFLSIM_COARSE_LATE") == "1"


def coarse_state(state: np.ndarray) -> np.ndarray:
    """Coarser late-game grouping used as a fallback for small pace cells:
    trailing big / trailing / tied / leading / leading big x (>2:00, <=2:00)."""
    state = np.asarray(state)
    sit = (state - 2) % len(LATE_SITS)
    t = (state - 2) // len(LATE_SITS)
    grp = np.array([0, 1, 1, 2, 3, 3, 4])[np.clip(sit, 0, 6)]
    return np.where(state < 2, state, 2 + (t == 2) * 5 + grp)


# --------------------------------------------------------------------------- #
# Elo
# --------------------------------------------------------------------------- #
@dataclass
class EloParams:
    k: float = 20.0
    revert: float = 1 / 3          # regression to the mean between seasons
    elo_per_point: float = 25.0
    hfa_window_seasons: int = 3    # HFA estimated from mean home margin of prior seasons


def compute_elo(games: pd.DataFrame, p: EloParams | None = None) -> pd.DataFrame:
    """Pregame Elo for every game, using only results from earlier dates."""
    p = p or EloParams()
    elo: dict[str, float] = {}
    last_season: dict[str, int] = {}
    played = []  # (season, home margin) for non-neutral games
    rows = []
    for day, dg in games.groupby("gameday", sort=True):
        for g in dg.itertuples():
            prior = [m for s, m in played if g.season - p.hfa_window_seasons <= s < g.season]
            hfa_pts = float(np.mean(prior)) if len(prior) > 100 else 2.0
            for t in (g.home_team, g.away_team):
                if t not in elo:
                    elo[t] = 1500.0
                if last_season.get(t, g.season) != g.season:
                    elo[t] = elo[t] * (1 - p.revert) + 1500.0 * p.revert
                last_season[t] = g.season
            hfa = 0.0 if g.neutral else hfa_pts * p.elo_per_point
            d = elo[g.home_team] - elo[g.away_team] + hfa
            rows.append({"game_id": g.game_id, "home_elo": elo[g.home_team],
                         "away_elo": elo[g.away_team], "elo_hfa": hfa,
                         "elo_wp": 1 / (1 + 10 ** (-d / 400))})
        for g, r in zip(dg.itertuples(), rows[-len(dg):]):
            if pd.isna(g.label_home_score):
                continue
            mov = g.label_home_score - g.label_away_score
            wp = r["elo_wp"]
            res = 1.0 if mov > 0 else 0.0 if mov < 0 else 0.5
            d = r["home_elo"] - r["away_elo"] + r["elo_hfa"]
            winner_d = d if mov > 0 else -d
            mult = np.log(abs(mov) + 1) * 2.2 / (winner_d * 0.001 + 2.2) if mov != 0 else 1.0
            delta = p.k * mult * (res - wp)
            elo[g.home_team] += delta
            elo[g.away_team] -= delta
            if not g.neutral:
                played.append((g.season, mov))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Drive outcome model
# --------------------------------------------------------------------------- #
FEATURES = ["off_epa", "off_sr", "def_epa", "def_sr", "qb_epa", "qb_delta", "elo_diff",
            "home", "rest_diff", "yl", "yl2", "str_x_yl", "hs_scaled", "diff14"] + \
           [f"st_{s}" for s in STATES[1:]]


def team_side_features(gf: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Static (pregame) per-offense features for home-offense and away-offense rows."""
    out = {}
    for off, de in (("home", "away"), ("away", "home")):
        sgn = 1 if off == "home" else -1
        x = pd.DataFrame({
            "game_id": gf["game_id"],
            "posteam": gf[f"{off}_team"],
            "off_epa": gf[f"{off}_off_epa"] - gf["lg_epa"],
            "off_sr": gf[f"{off}_off_sr"] - gf["lg_sr"],
            "def_epa": gf[f"{de}_def_epa"] - gf["lg_epa"],
            "def_sr": gf[f"{de}_def_sr"] - gf["lg_sr"],
            "qb_epa": gf[f"{off}_qb_epa"] - gf["lg_epa"],
            "qb_delta": gf[f"{off}_qb_delta"].fillna(0.0),
            "elo_diff": (gf[f"{off}_elo"] - gf[f"{de}_elo"]) / 100.0,
            "home": sgn * (1 - gf["neutral"]),
            "rest_diff": np.clip((gf[f"{off}_rest"] - gf[f"{de}_rest"]) / 7.0, -1.5, 1.5),
        })
        # Same offense with the backup QB in (used after a simulated starter exit):
        # backup rating, and qb_delta moved by the backup-minus-starter gap.
        if f"{off}_qb2_epa" in gf:
            x["qb_epa_b"] = gf[f"{off}_qb2_epa"] - gf["lg_epa"]
            x["qb_delta_b"] = x["qb_delta"] + (gf[f"{off}_qb2_epa"] - gf[f"{off}_qb_epa"])
        out[off] = x
    return out


def dynamic_features(static: np.ndarray, static_cols: list[str], yl, half, half_secs, diff):
    """Combine static matchup columns with drive-start state into the FEATURES matrix."""
    n = len(yl)
    yl = np.asarray(yl, dtype=float) / 100.0
    st = state_index(half, half_secs, diff)
    cols = {c: static[:, i] if static.ndim == 2 else np.full(n, static[i])
            for i, c in enumerate(static_cols)}
    cols["yl"] = yl
    cols["yl2"] = yl ** 2
    cols["str_x_yl"] = (cols["off_epa"] + cols["def_epa"]) * yl
    cols["hs_scaled"] = np.minimum(np.asarray(half_secs, dtype=float), 900.0) / 900.0
    cols["diff14"] = np.clip(np.asarray(diff, dtype=float), -21, 21) / 14.0
    for i, s in enumerate(STATES[1:], start=1):
        cols[f"st_{s}"] = (st == i).astype(float)
    return np.column_stack([cols[f] for f in FEATURES]), st


STATIC_COLS = ["off_epa", "off_sr", "def_epa", "def_sr", "qb_epa", "qb_delta", "elo_diff",
               "home", "rest_diff"]


@dataclass
class DriveModel:
    C: float = 1.0
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    W: np.ndarray | None = None
    b: np.ndarray | None = None
    classes_: list = field(default_factory=list)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        self.mean_ = X.mean(0)
        self.scale_ = X.std(0) + 1e-9
        Z = (X - self.mean_) / self.scale_
        lr = LogisticRegression(C=self.C, max_iter=2000)
        lr.fit(Z, y, sample_weight=sample_weight)
        # Expand to all OUTCOMES (a class absent from training gets -inf logit).
        W = np.zeros((len(OUTCOMES), X.shape[1]))
        b = np.full(len(OUTCOMES), -50.0)
        for j, c in enumerate(lr.classes_):
            W[c] = lr.coef_[j]
            b[c] = lr.intercept_[j]
        self.W, self.b, self.classes_ = W, b, list(lr.classes_)
        return self

    def predict_proba(self, X: np.ndarray, offset: np.ndarray | None = None) -> np.ndarray:
        """offset: optional [n, n_outcomes] additive log-odds (used by market anchoring)."""
        Z = (X - self.mean_) / self.scale_
        L = Z @ self.W.T + self.b
        if offset is not None:
            L = L + offset
        L -= L.max(1, keepdims=True)
        P = np.exp(L)
        return P / P.sum(1, keepdims=True)


# --------------------------------------------------------------------------- #
# Empirical tables
# --------------------------------------------------------------------------- #
MIN_CELL = 30


def start_bucket(yl) -> np.ndarray:
    """Field-position bucket of a drive start: 0 = opp 1-20 ... 4 = own 1-19."""
    return np.clip((np.asarray(yl, dtype=float) - 1) // 20, 0, 4).astype(int)


@dataclass
class PaceTable:
    """Empirical drive resampling. A sampled row is a whole historical drive:
    (plays, seconds, pass+run plays, dropbacks) plus its `drive_uid`, which
    players.py uses as the play-by-play template for that simulated drive.

    Cells are (outcome, game state, start field-position bucket), falling back
    to (outcome, state), then (outcome, coarse late state), then (outcome) when a
    cell has < MIN_CELL drives.
    """
    arr: np.ndarray      # [k, 4] plays, secs, n_pr, n_pass
    uid: np.ndarray      # [k] drive_uid
    cells: dict          # (outcome_idx, state_idx, bucket) -> row indices into arr

    @classmethod
    def fit(cls, drives: pd.DataFrame) -> "PaceTable":
        d = drives.dropna(subset=["plays", "duration"])
        st = state_index(d["half"].to_numpy(), d["half_seconds_remaining"].to_numpy(),
                         d["score_differential"].to_numpy())
        oi = d["result"].map(OUT_IDX).to_numpy()
        b = start_bucket(d["yardline_100"].to_numpy())
        arr = d[["plays", "duration", "n_plays_pr", "n_pass_pr"]].to_numpy(float)
        uid = d["drive_uid"].to_numpy() if "drive_uid" in d else np.full(len(d), -1)
        cs = coarse_state(st)
        cells = {}
        for o in range(len(OUTCOMES)):
            pooled = np.flatnonzero(oi == o)
            for s in range(len(STATES)):
                ms = np.flatnonzero((oi == o) & (st == s))
                mc = np.flatnonzero((oi == o) & (cs == coarse_state(np.array([s]))[0]))
                fallback = ms if len(ms) >= MIN_CELL else (mc if len(mc) >= MIN_CELL else pooled)
                for bb in range(5):
                    m = np.flatnonzero((oi == o) & (st == s) & (b == bb))
                    cells[(o, s, bb)] = m if len(m) >= MIN_CELL else fallback
        return cls(arr, uid, cells)

    def sample(self, rng, outcome: np.ndarray, state: np.ndarray, start_yl: np.ndarray,
               u: np.ndarray | None = None):
        """Returns ([n, 4] plays/secs/n_pr/n_pass, [n] drive_uid).
        u: optional per-row uniforms (common random numbers); else drawn from rng."""
        u = rng.random(len(outcome)) if u is None else u
        idx = np.zeros(len(outcome), dtype=int)
        key = (outcome * 100 + state) * 10 + start_bucket(start_yl)
        for k in np.unique(key):
            m = key == k
            cell = self.cells[(k // 1000, (k // 10) % 100, k % 10)]
            idx[m] = cell[np.minimum((u[m] * len(cell)).astype(int), len(cell) - 1)]
        return self.arr[idx], self.uid[idx]


@dataclass
class FieldPosTable:
    cells: dict      # (outcome_idx, start_bucket) -> array of next-drive start yardline_100
    kickoff: np.ndarray

    @classmethod
    def fit(cls, drives: pd.DataFrame, n_recent_kicks: int = 1200) -> "FieldPosTable":
        d = drives.dropna(subset=["next_start_yl"]).sort_values("game_date")
        oi = d["result"].map(OUT_IDX).to_numpy()
        b = np.clip(d["yardline_100"].to_numpy() // 20, 0, 4).astype(int)
        nx = d["next_start_yl"].to_numpy()
        is_kick = np.isin(oi, [OUT_IDX["TD"], OUT_IDX["FG"], OUT_IDX["OPP_TD"]])
        kick = nx[is_kick][-n_recent_kicks:]
        cells = {}
        for o in range(len(OUTCOMES)):
            pooled = nx[oi == o]
            for bb in range(5):
                m = (oi == o) & (b == bb)
                cells[(o, bb)] = nx[m] if m.sum() >= MIN_CELL else (pooled if len(pooled) else kick)
        for o in ("TD", "FG", "OPP_TD"):
            for bb in range(5):
                cells[(OUT_IDX[o], bb)] = kick
        return cls(cells, kick)

    def sample_kick(self, rng, n: int, u: np.ndarray | None = None) -> np.ndarray:
        u = rng.random(n) if u is None else u
        k = len(self.kickoff)
        return self.kickoff[np.minimum((u * k).astype(int), k - 1)]

    def sample(self, rng, outcome: np.ndarray, start_yl: np.ndarray,
               u: np.ndarray | None = None) -> np.ndarray:
        u = rng.random(len(outcome)) if u is None else u
        b = np.clip(np.asarray(start_yl) // 20, 0, 4).astype(int)
        key = outcome * 10 + b
        out = np.zeros(len(outcome))
        for k in np.unique(key):
            m = key == k
            cell = self.cells[(k // 10, k % 10)]
            out[m] = cell[np.minimum((u[m] * len(cell)).astype(int), len(cell) - 1)]
        return out


@dataclass
class PATModel:
    go2: dict          # (late, diff_after_td clipped to [-20, 20]) -> P(go for 2)
    go2_default: float
    xp_rate: float
    two_rate: float

    @classmethod
    def fit(cls, pat: pd.DataFrame) -> "PATModel":
        late = (pat["qtr"] >= 4).astype(int)
        diff = pat["score_differential"].clip(-20, 20).astype(int)
        g = pat.assign(late=late, diff=diff).groupby(["late", "diff"])["go2"].agg(["mean", "size"])
        go2 = {k: r["mean"] for k, r in g.iterrows() if r["size"] >= 20}
        return cls(go2, float(pat["go2"].mean()),
                   float(pat.loc[pat.go2 == 0, "good"].mean()),
                   float(pat.loc[pat.go2 == 1, "good"].mean()))

    def points(self, rng, late: np.ndarray, diff_after_td: np.ndarray,
               u: np.ndarray | None = None) -> np.ndarray:
        """u: optional [n, 2] uniforms (go-for-two decision, success)."""
        u = rng.random((len(diff_after_td), 2)) if u is None else u
        d = np.clip(diff_after_td, -20, 20).astype(int)
        p2 = np.array([self.go2.get((int(l), int(x)), self.go2_default) for l, x in zip(late, d)])
        go2 = u[:, 0] < p2
        good = u[:, 1] < np.where(go2, self.two_rate, self.xp_rate)
        return np.where(good, np.where(go2, 2, 1), 0)


# --------------------------------------------------------------------------- #
# Bundle
# --------------------------------------------------------------------------- #
@dataclass
class TeamModel:
    drive: DriveModel
    pace: PaceTable
    fieldpos: FieldPosTable
    pat: PATModel
    cutoff: pd.Timestamp
    n_train_drives: int


def drive_training_matrix(drives: pd.DataFrame, gf: pd.DataFrame):
    sides = team_side_features(gf)
    stat = pd.concat([sides["home"], sides["away"]])
    d = drives[drives["half"] <= 2].merge(stat, on=["game_id", "posteam"], how="inner")
    X, _ = dynamic_features(d[STATIC_COLS].to_numpy(float), STATIC_COLS,
                            d["yardline_100"].to_numpy(), d["half"].to_numpy(),
                            d["half_seconds_remaining"].to_numpy(),
                            d["score_differential"].to_numpy())
    y = d["result"].map(OUT_IDX).to_numpy()
    ok = np.isfinite(X).all(1)
    return X[ok], y[ok], d[ok]


def with_elo(gf: pd.DataFrame, elo: pd.DataFrame) -> pd.DataFrame:
    gf = gf.merge(elo[["game_id", "home_elo", "away_elo", "elo_wp"]], on="game_id", how="left")
    return gf


def fit_team_model(cutoff, drives: pd.DataFrame, pat: pd.DataFrame, gf: pd.DataFrame,
                   train_seasons: int = 4, pace_seasons: int = 3, C: float = 1.0,
                   season_decay: float = 0.7,
                   rng: np.random.Generator | None = None, bootstrap: bool = False) -> TeamModel:
    """Fit on data strictly before `cutoff`. gf must already include Elo columns.

    season_decay < 1 down-weights older seasons in the drive model (tracks
    league-wide scoring drift).
    """
    cutoff = pd.Timestamp(cutoff)
    d = drives[drives["game_date"] < cutoff]
    last_season = d["season"].max()
    dtrain = d[d["season"] > last_season - train_seasons]
    if bootstrap:  # resample whole games (used for model-uncertainty estimates)
        rng = rng or np.random.default_rng()
        gids = dtrain["game_id"].unique()
        pick = rng.choice(gids, len(gids), replace=True)
        cnt = pd.Series(pick).value_counts()
        dtrain = dtrain[dtrain["game_id"].isin(cnt.index)]
        w_game = cnt
    X, y, dd = drive_training_matrix(dtrain, gf)
    sw = dd["game_id"].map(w_game).to_numpy(float) if bootstrap else np.ones(len(y))
    # Recency weighting: a drive from k seasons before the latest season gets season_decay**k.
    sw = sw * season_decay ** (last_season - dd["season"].to_numpy())
    dm = DriveModel(C=C).fit(X, y, sw)
    dpace = d[(d["season"] > last_season - pace_seasons) & (d["half"] <= 2)]
    p = pat[(pat["game_date"] < cutoff) & (pat["season"] > last_season - pace_seasons)]
    return TeamModel(dm, PaceTable.fit(dpace), FieldPosTable.fit(dpace), PATModel.fit(p),
                     cutoff, len(y))


def expected_points_per_drive(tm: TeamModel, static_row: np.ndarray, n: int = 4000,
                              seed: int = 0) -> dict:
    """Expected points and outcome probabilities for a neutral-state drive
    (1st half, 15:00 left, tied) starting from the recent kickoff distribution."""
    rng = np.random.default_rng(seed)
    yl = tm.fieldpos.sample_kick(rng, n)
    X, _ = dynamic_features(np.asarray(static_row, float), STATIC_COLS, yl,
                            np.ones(n, int), np.full(n, 900.0), np.zeros(n))
    P = tm.drive.predict_proba(X).mean(0)
    pts = {"TD": 6 + tm.pat.xp_rate, "FG": 3, "OPP_TD": -(6 + tm.pat.xp_rate), "SAFETY": -2}
    ep = sum(P[OUT_IDX[k]] * v for k, v in pts.items())
    return {"ep_per_drive": float(ep), **{o: float(P[i]) for i, o in enumerate(OUTCOMES)}}
