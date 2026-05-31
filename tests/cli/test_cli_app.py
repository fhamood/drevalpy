"""Tests for the Typer-based ``drevalpy`` CLI."""

from __future__ import annotations

import re
import warnings

import pytest
from typer.testing import CliRunner

from drevalpy.cli.legacy import load_response
from drevalpy.cli.main import app

runner = CliRunner()
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _plain_stdout(text: str) -> str:
    """Strip terminal escape codes (e.g. when CI sets ``FORCE_COLOR=1``)."""
    return _ANSI_ESCAPE.sub("", text)


def test_drevalpy_help_lists_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "viability-preprocess" in result.stdout
    assert "train-cv" in result.stdout
    assert "make-pipeline-report" in result.stdout


def test_viability_preprocess_help() -> None:
    result = runner.invoke(
        app,
        ["viability-preprocess", "--help"],
        env={"FORCE_COLOR": "1", "CI": "true"},
    )
    assert result.exit_code == 0
    help_text = _plain_stdout(result.stdout)
    assert "--dataset_name" in help_text
    assert "--path_data" in help_text
    assert "--cores" in help_text


def test_load_response_requires_response_dataset() -> None:
    result = runner.invoke(app, ["load-response"])
    assert result.exit_code != 0


def test_legacy_load_response_emits_deprecation_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy script warns and forwards argv to the Typer subcommand."""
    monkeypatch.setattr("sys.argv", ["drevalpy-load-response", "--help"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(SystemExit) as exc_info:
            load_response()

    assert exc_info.value.code == 0
    assert any(
        issubclass(w.category, DeprecationWarning) and "drevalpy load-response" in str(w.message) for w in caught
    )


def test_pipeline_root_missing_models_fails_fast() -> None:
    result = runner.invoke(
        app,
        [
            "--dataset_name",
            "GDSC1",
        ],
    )
    assert result.exit_code != 0
    assert result.exception is not None
    assert "At least one model must be specified" in str(result.exception)
