"""Small helpers shared by Typer command modules."""

from __future__ import annotations

from argparse import Namespace
from typing import Any


def as_list(value: list[str] | tuple[str, ...] | None) -> list[str]:
    """Normalize Typer list options to a plain list."""
    if value is None:
        return []
    return list(value)


def pipeline_namespace(**kwargs: Any) -> Namespace:
    """Build an ``argparse.Namespace`` for the full-pipeline entry point."""
    return Namespace(**kwargs)
