"""Shared test setup."""
import pytest


@pytest.fixture(autouse=True)
def _no_live_ask_page(monkeypatch):
    """Hide the live "Ask the Archive" page in every test unless a test turns it on.

    The page appears only where the documents, indexes and an API key exist, so without this the
    dashboard tests would see six pages on a fresh clone and seven on a machine with data.
    """
    from driftmon.dashboard import ask
    monkeypatch.setattr(ask, "missing", lambda cfg: ["(hidden in tests)"])
