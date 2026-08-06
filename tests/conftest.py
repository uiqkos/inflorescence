"""Shared fixtures.

The isolation below is not a convenience: `init` writes the API key to a per-user config
directory, and the first run of its test suite wrote `sk-test` into the developer's real
~/.config/inflorescence/.env. A test must not be able to reach outside the sandbox to do
that, so the redirect is autouse and applies to every test in the suite.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_user_config(tmp_path_factory, monkeypatch):
    """Point the per-user config directory at a temporary path for every test."""
    home = tmp_path_factory.mktemp("user-config")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    monkeypatch.setenv("APPDATA", str(home))
    return home
