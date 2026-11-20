"""Test-wide settings."""

from hypothesis import settings

# Property tests run on shared machines. A per-example deadline would let
# machine load decide whether a test passes.
settings.register_profile("no_deadline", deadline=None)
settings.load_profile("no_deadline")
