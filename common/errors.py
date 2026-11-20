"""Exception types shared by more than one package."""

from __future__ import annotations


class ConfigError(Exception):
    """A configuration value is missing, malformed or inconsistent. Raised at startup."""
