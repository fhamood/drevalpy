"""Small helpers shared by Typer command modules."""

from __future__ import annotations

from argparse import Namespace
from typing import Any

LIST_OPTIONS = frozenset(
    {
        "--models",
        "--baselines",
        "--test_mode",
        "--randomization_mode",
        "--cross_study_datasets",
        "--test_modes",
        "--hpam_yamls",
        "--pred_datas",
        "--outfiles",
    }
)


def _is_option_token(token: str) -> bool:
    """Return whether *token* begins a new CLI option."""
    return token.startswith("-") and token not in {"-", "--"}


def normalize_list_argv(argv: list[str], list_options: frozenset[str] = LIST_OPTIONS) -> list[str]:
    """Expand argparse-style space-separated list options for Typer/Click.

    Converts ``--models A B`` into ``--models A --models B`` while leaving
    repeated-flag syntax unchanged.

    :param argv: Command-line tokens without the program name.
    :param list_options: Option names that accept one or more trailing values.
    :return: Normalized argv suitable for Typer/Click list options.
    """
    normalized: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token not in list_options:
            normalized.append(token)
            index += 1
            continue

        index += 1
        values: list[str] = []
        while index < len(argv) and not _is_option_token(argv[index]):
            values.append(argv[index])
            index += 1

        if not values:
            normalized.append(token)
            continue

        for value in values:
            normalized.extend([token, value])

    return normalized


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
