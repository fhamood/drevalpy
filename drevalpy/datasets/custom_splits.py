"""User-provided split creation for advanced CV setups (issue #407)."""

from __future__ import annotations

import csv
import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..pipeline_function import pipeline_function
from .dataset import DrugResponseDataset, split_early_stopping_data

TEST_MODES: frozenset[str] = frozenset({"LPO", "LCO", "LDO", "LTO"})
REQUIRED_ROLES: tuple[str, ...] = ("train", "validation", "test")
OPTIONAL_ROLES: tuple[str, ...] = ("validation_es", "early_stopping")
MANIFEST_FILENAME = "split_manifest.csv"


def validate_split_label(label: str) -> str:
    """
    Ensure a custom result-directory label is safe for paths and report parsing.

    :param label: directory name used under the dataset results folder
    :returns: the validated label unchanged
    :raises CustomSplitError: if the label is empty or contains path separators
    """
    if not label or label.strip() != label:
        msg = "split label must be a non-empty string without leading or trailing whitespace"
        raise CustomSplitError(msg)
    if "/" in label or "\\" in label:
        msg = f"split label must not contain path separators: {label!r}"
        raise CustomSplitError(msg)
    return label


class CustomSplitError(ValueError):
    """Raised when a custom split script or its output is invalid."""


CustomSplitCreator = Callable[[DrugResponseDataset], list[dict[str, DrugResponseDataset]]]


def load_custom_splitter(path: Path | str) -> CustomSplitCreator:
    """
    Load a module-level ``create_splits`` function from a Python script.

    :param path: path to a Python file defining ``create_splits(response_data)``
    :returns: the loaded splitter callable
    :raises FileNotFoundError: if the script path does not exist
    :raises ImportError: if the script cannot be imported
    :raises AttributeError: if ``create_splits`` is missing
    :raises TypeError: if ``create_splits`` is not callable
    """
    script_path = Path(path).expanduser().resolve()
    if not script_path.is_file():
        msg = f"Custom split script not found: {script_path}"
        raise FileNotFoundError(msg)

    spec = importlib.util.spec_from_file_location("_drevalpy_custom_split_", script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load custom split script: {script_path}"
        raise ImportError(msg)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "create_splits", None)
    if fn is None:
        msg = f"{script_path} must define a module-level function create_splits(response_data)"
        raise AttributeError(msg)
    if not callable(fn):
        msg = "create_splits must be callable"
        raise TypeError(msg)
    return fn  # type: ignore[return-value]


def _row_keys(dataset: DrugResponseDataset) -> set[tuple[str, str, float]]:
    return {
        (str(cl), str(drug), float(resp))
        for cl, drug, resp in zip(dataset.cell_line_ids, dataset.drug_ids, dataset.response, strict=True)
    }


def _group_ids(dataset: DrugResponseDataset, test_mode: str) -> set[Any]:
    if test_mode == "LCO":
        return set(map(str, dataset.cell_line_ids))
    if test_mode == "LDO":
        return set(map(str, dataset.drug_ids))
    if test_mode == "LTO":
        if dataset.tissue is None:
            msg = "LTO validation requires tissue annotations on all role datasets"
            raise CustomSplitError(msg)
        return set(map(str, dataset.tissue))
    if test_mode == "LPO":
        return {(str(cl), str(drug)) for cl, drug in zip(dataset.cell_line_ids, dataset.drug_ids, strict=True)}
    msg = f"Unknown test_mode {test_mode!r}; choose from {sorted(TEST_MODES)}"
    raise CustomSplitError(msg)


def _assert_disjoint_groups(
    roles: dict[str, DrugResponseDataset],
    test_mode: str,
    *,
    split_index: int,
) -> None:
    seen: dict[Any, str] = {}
    for role in REQUIRED_ROLES:
        for group in _group_ids(roles[role], test_mode):
            if group in seen:
                msg = (
                    f"Split {split_index}: {test_mode} leakage — " f"{seen[group]!r} and {role!r} share group {group!r}"
                )
                raise CustomSplitError(msg)
            seen[group] = role


def _assert_disjoint_rows(
    roles: dict[str, DrugResponseDataset],
    *,
    split_index: int,
) -> None:
    seen: dict[tuple[str, str, float], str] = {}
    for role in REQUIRED_ROLES:
        for row in _row_keys(roles[role]):
            if row in seen:
                msg = f"Split {split_index}: exact row overlap between " f"{seen[row]!r} and {role!r}: {row[:2]}"
                raise CustomSplitError(msg)
            seen[row] = role


def _normalize_split_dict(
    raw: dict[str, Any], *, split_index: int
) -> tuple[dict[str, DrugResponseDataset], dict[str, Any]]:
    metadata = raw.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        msg = f"Split {split_index}: metadata must be a dict when provided"
        raise CustomSplitError(msg)

    roles: dict[str, DrugResponseDataset] = {}
    for role in REQUIRED_ROLES + OPTIONAL_ROLES:
        if role not in raw:
            continue
        value = raw[role]
        if not isinstance(value, DrugResponseDataset):
            msg = f"Split {split_index}: {role!r} must be a DrugResponseDataset, got {type(value).__name__}"
            raise CustomSplitError(msg)
        roles[role] = value

    for role in REQUIRED_ROLES:
        if role not in roles:
            msg = f"Split {split_index}: missing required role {role!r}"
            raise CustomSplitError(msg)
        if len(roles[role]) == 0:
            msg = f"Split {split_index}: role {role!r} must not be empty"
            raise CustomSplitError(msg)

    dataset_names = {roles[role].dataset_name for role in REQUIRED_ROLES}
    if len(dataset_names) != 1:
        msg = f"Split {split_index}: inconsistent dataset_name across roles: {dataset_names}"
        raise CustomSplitError(msg)

    return roles, metadata if isinstance(metadata, dict) else {}


@pipeline_function
def validate_cv_splits(
    splits: list[dict[str, Any]],
    test_mode: str,
) -> tuple[list[dict[str, DrugResponseDataset]], list[dict[str, Any]]]:
    """
    Validate custom split output according to ``test_mode`` semantics.

    :param splits: raw split dicts returned by a custom splitter
    :param test_mode: one of ``LPO``, ``LCO``, ``LDO``, or ``LTO``
    :returns: validated role datasets and optional per-split metadata rows
    :raises CustomSplitError: if splits are missing roles or leak across groups/rows
    """
    if test_mode not in TEST_MODES:
        msg = f"Unknown test_mode {test_mode!r}; choose from {sorted(TEST_MODES)}"
        raise CustomSplitError(msg)
    if not splits:
        msg = "Custom splitter returned no splits"
        raise CustomSplitError(msg)

    validated: list[dict[str, DrugResponseDataset]] = []
    metadata_rows: list[dict[str, Any]] = []
    for split_index, raw in enumerate(splits):
        if not isinstance(raw, dict):
            msg = f"Split {split_index}: expected dict, got {type(raw).__name__}"
            raise CustomSplitError(msg)
        roles, metadata = _normalize_split_dict(raw, split_index=split_index)
        _assert_disjoint_rows(roles, split_index=split_index)
        _assert_disjoint_groups(roles, test_mode, split_index=split_index)
        validated.append(roles)
        metadata_rows.append({"split_index": split_index, **metadata})

    return validated, metadata_rows


@pipeline_function
def ensure_early_stopping_splits(
    splits: list[dict[str, DrugResponseDataset]],
    test_mode: str,
) -> None:
    """
    Fill ``validation_es`` and ``early_stopping`` when absent.

    :param splits: validated split dicts to mutate in place
    :param test_mode: one of ``LPO``, ``LCO``, ``LDO``, or ``LTO``
    """
    for split in splits:
        if "validation_es" in split and "early_stopping" in split:
            continue
        validation = split["validation"]
        n_groups = len(_group_ids(validation, test_mode))
        if n_groups < 2:
            split["validation_es"] = validation.copy()
            split["early_stopping"] = DrugResponseDataset(
                response=np.array([]),
                cell_line_ids=np.array([]),
                drug_ids=np.array([]),
                tissues=np.array([]) if validation.tissue is not None else None,
                dataset_name=validation.dataset_name,
            )
            continue
        validation_es, early_stopping = split_early_stopping_data(validation, test_mode=test_mode)
        split["validation_es"] = validation_es
        split["early_stopping"] = early_stopping


def write_split_manifest(path: Path | str, metadata_rows: list[dict[str, Any]], test_mode: str) -> None:
    """
    Write optional split metadata next to persisted split files.

    :param path: directory where split CSV files are stored
    :param metadata_rows: per-split metadata collected during validation
    :param test_mode: validation mode recorded in the manifest
    """
    if not metadata_rows:
        return
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in metadata_rows for key in row.keys()} | {"test_mode"})
    manifest_path = out / MANIFEST_FILENAME
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metadata_rows:
            writer.writerow({"test_mode": test_mode, **row})


@pipeline_function
def run_custom_splitter(
    response_data: DrugResponseDataset,
    splitter: CustomSplitCreator | str | Path,
    *,
    test_mode: str,
    split_early_stopping: bool = True,
) -> tuple[list[dict[str, DrugResponseDataset]], list[dict[str, Any]]]:
    """
    Execute a custom splitter, validate output, and normalize early-stopping roles.

    :param response_data: full response dataset passed to the splitter
    :param splitter: callable or path to a script defining ``create_splits``
    :param test_mode: one of ``LPO``, ``LCO``, ``LDO``, or ``LTO``
    :param split_early_stopping: whether to derive early-stopping roles when absent
    :returns: validated splits and optional metadata rows
    """
    if isinstance(splitter, (str, Path)):
        splitter = load_custom_splitter(splitter)

    raw_splits = splitter(response_data.copy())
    training_splits, metadata_rows = validate_cv_splits(raw_splits, test_mode)
    if split_early_stopping:
        ensure_early_stopping_splits(training_splits, test_mode)
    return training_splits, metadata_rows
