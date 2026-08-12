"""Packaging invariants that don't fit test_client.py — a PEP 561 marker present on disk (proof
the wheel build actually included it is a manual `unzip -l dist/*.whl` step, not part of CI)."""

from __future__ import annotations

from pathlib import Path

import bggpt_toolkit


def test_py_typed_marker_is_present() -> None:
    assert (Path(bggpt_toolkit.__file__).parent / "py.typed").is_file()
