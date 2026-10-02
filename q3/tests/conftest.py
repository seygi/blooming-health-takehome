from pathlib import Path

import pytest

from callcheck.load import load

DATA = Path(__file__).resolve().parents[1] / "data" / "gym_agent_conversations.json"


@pytest.fixture(scope="session")
def loaded():
    return load(DATA)


@pytest.fixture(scope="session")
def spec(loaded):
    return loaded[0]


@pytest.fixture(scope="session")
def threads(loaded):
    return loaded[1]
