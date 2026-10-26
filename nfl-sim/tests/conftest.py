import pytest

from nflsim.ingest import PROCESSED


@pytest.fixture(scope="session")
def data():
    if not (PROCESSED / "game_features.parquet").exists():
        pytest.skip("processed data missing: run `python -m nflsim.ingest` and `python -m nflsim.features`")
    from nflsim.features import read_processed
    from nflsim.team_model import compute_elo, with_elo
    games = read_processed("games")
    gf = with_elo(read_processed("game_features"), compute_elo(games))
    return {"games": games, "gf": gf, "drives": read_processed("drives"),
            "pat": read_processed("pat"), "team_games": read_processed("team_games"),
            "qb_games": read_processed("qb_games")}


@pytest.fixture(scope="session")
def model(data):
    from nflsim.team_model import fit_team_model
    return fit_team_model("2023-09-01", data["drives"], data["pat"], data["gf"])


@pytest.fixture(scope="session")
def matchup(data):
    from nflsim.simulator import matchup_from_features
    row = data["gf"][data["gf"]["game_id"] == "2023_01_DET_KC"].iloc[0]
    return matchup_from_features(row)
