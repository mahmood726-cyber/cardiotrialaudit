"""Shared pytest helpers for cardiotrialaudit.

The full-pipeline tests (``build_master_table`` / ``filter_cardiology_trials``)
read a multi-gigabyte AACT ClinicalTrials.gov ZIP snapshot from disk. That
snapshot is an external integration dependency, not vendored with the repo, so
its path is supplied via the ``CARDIOTRIALAUDIT_AACT_ZIP`` environment variable
(see ``pipeline/ingest.py``). When it is absent these integration tests are
skipped with a clear reason rather than failing — the unit tests that use the
in-repo fixture ZIP still run.
"""
import functools

import pytest


def skip_if_no_aact(func):
    """Skip an integration test when the AACT snapshot is unavailable."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as exc:
            pytest.skip(f"AACT snapshot not available: {exc}")

    return wrapper
