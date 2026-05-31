"""Small helpers shared by Typer command modules."""

from __future__ import annotations

from argparse import Namespace
from typing import Any


def as_list(value: list[str] | tuple[str, ...] | None) -> list[str]:
    """Normalize Typer list options to a plain list.

    :param value: Optional sequence from a Typer multi-value option.
    :return: A plain list (empty when *value* is ``None``).
    """
    if value is None:
        return []
    return list(value)


def pipeline_namespace(**kwargs: Any) -> Namespace:
    """Build an ``argparse.Namespace`` for the full-pipeline entry point.

    :param kwargs: Pipeline option names and values.
    :return: Namespace consumed by ``drevalpy.utils.main``.
    """
    return Namespace(**kwargs)
