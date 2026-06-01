"""Tests for :mod:`drevalpy.cli._helpers`."""

from __future__ import annotations

from drevalpy.cli._helpers import normalize_list_argv


def test_normalize_list_argv_expands_space_separated_values() -> None:
    argv = ["--models", "Simple", "ElasticNet", "--dataset_name", "GDSC1"]
    assert normalize_list_argv(argv) == [
        "--models",
        "Simple",
        "--models",
        "ElasticNet",
        "--dataset_name",
        "GDSC1",
    ]


def test_normalize_list_argv_preserves_repeated_flags() -> None:
    argv = ["evaluate-hpams", "--hpam_yamls", "a.yml", "--hpam_yamls", "b.yml"]
    assert normalize_list_argv(argv) == [
        "evaluate-hpams",
        "--hpam_yamls",
        "a.yml",
        "--hpam_yamls",
        "b.yml",
    ]


def test_normalize_list_argv_stops_at_next_option() -> None:
    argv = ["--test_mode", "LPO", "LCO", "--path_out", "results/"]
    assert normalize_list_argv(argv) == [
        "--test_mode",
        "LPO",
        "--test_mode",
        "LCO",
        "--path_out",
        "results/",
    ]
