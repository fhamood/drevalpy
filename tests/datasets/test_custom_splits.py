"""Tests for drevalpy.datasets.custom_splits."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from drevalpy.datasets.custom_splits import (
    CustomSplitError,
    load_custom_splitter,
    run_custom_splitter,
    validate_cv_splits,
    validate_split_label,
    write_split_manifest,
)
from drevalpy.datasets.dataset import DrugResponseDataset


def _sample_dataset(n_cell_lines: int = 6, n_drugs: int = 4) -> DrugResponseDataset:
    cell_lines = np.repeat([f"CL-{i}" for i in range(n_cell_lines)], n_drugs)
    drugs = np.tile([f"D-{i}" for i in range(n_drugs)], n_cell_lines)
    tissues = np.repeat([f"T-{i % 3}" for i in range(n_cell_lines)], n_drugs)
    return DrugResponseDataset(
        response=np.random.default_rng(0).random(len(cell_lines)),
        cell_line_ids=cell_lines,
        drug_ids=drugs,
        tissues=tissues,
        dataset_name="testset",
    )


def _role_from_groups(
    dataset: DrugResponseDataset,
    *,
    train_groups: set[str],
    val_groups: set[str],
    test_groups: set[str],
    group_col: str,
) -> dict[str, DrugResponseDataset]:
    if group_col == "cell_line":
        groups = dataset.cell_line_ids
    elif group_col == "drug":
        groups = dataset.drug_ids
    else:
        assert dataset.tissue is not None
        groups = dataset.tissue

    def subset(selected: set[str]) -> DrugResponseDataset:
        mask = np.isin(groups, list(selected))
        return DrugResponseDataset(
            response=dataset.response[mask],
            cell_line_ids=dataset.cell_line_ids[mask],
            drug_ids=dataset.drug_ids[mask],
            tissues=dataset.tissue[mask] if dataset.tissue is not None else None,
            dataset_name=dataset.dataset_name,
        )

    return {
        "train": subset(train_groups),
        "validation": subset(val_groups),
        "test": subset(test_groups),
    }


def test_validate_split_label_rejects_path_separators() -> None:
    """Reject split labels that contain path separators."""
    with pytest.raises(CustomSplitError):
        validate_split_label("scaling/lco")


def test_load_custom_splitter_requires_create_splits(tmp_path: Path) -> None:
    """
    Require a module-level create_splits function in custom scripts.

    :param tmp_path: Temporary path provided by pytest.
    """
    script = tmp_path / "bad.py"
    script.write_text("def other():\n    pass\n", encoding="utf-8")
    with pytest.raises(AttributeError, match="create_splits"):
        load_custom_splitter(script)


def test_validate_lco_rejects_shared_cell_line() -> None:
    """Reject LCO splits that share cell lines across roles."""
    dataset = _sample_dataset()
    split = _role_from_groups(
        dataset,
        train_groups={"CL-0", "CL-1"},
        val_groups={"CL-1", "CL-2"},
        test_groups={"CL-3"},
        group_col="cell_line",
    )
    with pytest.raises(CustomSplitError, match="overlap|leakage"):
        validate_cv_splits([split], "LCO")


def test_validate_ldo_rejects_shared_drug() -> None:
    """Reject LDO splits that share drugs across roles."""
    dataset = _sample_dataset()
    split = _role_from_groups(
        dataset,
        train_groups={"D-0", "D-1"},
        val_groups={"D-1", "D-2"},
        test_groups={"D-3"},
        group_col="drug",
    )
    with pytest.raises(CustomSplitError, match="overlap|leakage"):
        validate_cv_splits([split], "LDO")


def test_validate_lto_requires_tissue_and_disjointness() -> None:
    """Accept valid LTO splits with disjoint tissue groups."""
    dataset = _sample_dataset()
    split = _role_from_groups(
        dataset,
        train_groups={"T-0"},
        val_groups={"T-1"},
        test_groups={"T-2"},
        group_col="tissue",
    )
    validated, metadata = validate_cv_splits([split], "LTO")
    assert len(validated) == 1
    assert metadata[0]["split_index"] == 0


def test_validate_lpo_rejects_shared_pair() -> None:
    """Reject LPO splits that share cell-line/drug pairs across roles."""
    split = {
        "train": DrugResponseDataset(
            response=np.array([1.0]),
            cell_line_ids=np.array(["CL-0"]),
            drug_ids=np.array(["D-0"]),
            dataset_name="testset",
        ),
        "validation": DrugResponseDataset(
            response=np.array([2.0]),
            cell_line_ids=np.array(["CL-0"]),
            drug_ids=np.array(["D-0"]),
            dataset_name="testset",
        ),
        "test": DrugResponseDataset(
            response=np.array([3.0]),
            cell_line_ids=np.array(["CL-1"]),
            drug_ids=np.array(["D-1"]),
            dataset_name="testset",
        ),
    }
    with pytest.raises(CustomSplitError, match="LPO leakage"):
        validate_cv_splits([split], "LPO")


def test_run_custom_splitter_adds_early_stopping_roles(tmp_path: Path) -> None:
    """
    Add early-stopping roles when running a custom splitter script.

    :param tmp_path: Temporary path provided by pytest.
    """
    script = tmp_path / "splitter.py"
    script.write_text(
        """
import numpy as np
from drevalpy.datasets.dataset import DrugResponseDataset

def create_splits(response_data, params):
    mask_train = response_data.cell_line_ids == "CL-0"
    mask_val = response_data.cell_line_ids == "CL-1"
    mask_test = response_data.cell_line_ids == "CL-2"
    def pick(mask):
        return DrugResponseDataset(
            response=response_data.response[mask],
            cell_line_ids=response_data.cell_line_ids[mask],
            drug_ids=response_data.drug_ids[mask],
            tissues=response_data.tissue[mask],
            dataset_name=response_data.dataset_name,
        )
    return [{"train": pick(mask_train), "validation": pick(mask_val), "test": pick(mask_test)}]
""",
        encoding="utf-8",
    )
    dataset = _sample_dataset(n_cell_lines=3, n_drugs=2)
    splits, metadata = run_custom_splitter(
        dataset,
        script,
        test_mode="LCO",
        random_state=7,
        validation_ratio=0.25,
    )
    assert "validation_es" in splits[0]
    assert "early_stopping" in splits[0]
    assert metadata[0]["split_index"] == 0


def test_write_split_manifest(tmp_path: Path) -> None:
    """
    Write split metadata to split_manifest.csv.

    :param tmp_path: Temporary path provided by pytest.
    """
    write_split_manifest(tmp_path, [{"split_index": 0, "fraction": 0.5}], "LCO")
    manifest = (tmp_path / "split_manifest.csv").read_text(encoding="utf-8")
    assert "fraction" in manifest
    assert "LCO" in manifest


def test_make_cv_pkls_with_custom_splitter(tmp_path: Path) -> None:
    """
    Generate split pickle files from a custom splitter via run_cv_split.

    :param tmp_path: Temporary path provided by pytest.
    """
    from drevalpy.cli_run_cv import run_cv_split

    dataset = _sample_dataset(n_cell_lines=4, n_drugs=2)
    response_pkl = tmp_path / "response.pkl"
    with response_pkl.open("wb") as handle:
        pickle.dump(dataset, handle)

    script = tmp_path / "splitter.py"
    script.write_text(
        """
from drevalpy.datasets.dataset import DrugResponseDataset

def create_splits(response_data, params):
    train = response_data.cell_line_ids == "CL-0"
    val = response_data.cell_line_ids == "CL-1"
    test = response_data.cell_line_ids == "CL-2"
    def subset(mask):
        return DrugResponseDataset(
            response=response_data.response[mask],
            cell_line_ids=response_data.cell_line_ids[mask],
            drug_ids=response_data.drug_ids[mask],
            tissues=response_data.tissue[mask],
            dataset_name=response_data.dataset_name,
        )
    return [
        {"train": subset(train), "validation": subset(val), "test": subset(test)},
        {
            "train": subset(response_data.cell_line_ids == "CL-3"),
            "validation": subset(val),
            "test": subset(test),
        },
    ]
""",
        encoding="utf-8",
    )

    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        run_cv_split(
            response=str(response_pkl.name),
            n_cv_splits=2,
            test_mode="LCO",
            custom_splitter_path=str(script),
        )
        assert (tmp_path / "split_0.pkl").is_file()
        assert (tmp_path / "split_1.pkl").is_file()
        with (tmp_path / "split_0.pkl").open("rb") as handle:
            split = pickle.load(handle)
        assert {"train", "validation", "test", "validation_es", "early_stopping"}.issubset(split)
    finally:
        os.chdir(cwd)


def test_run_custom_splitter_forwards_params_to_script(tmp_path: Path) -> None:
    """
    Forward CustomSplitParams fields to create_splits scripts.

    :param tmp_path: Temporary path provided by pytest.
    """
    script = tmp_path / "params_splitter.py"
    script.write_text(
        """
from drevalpy.datasets.dataset import DrugResponseDataset

def create_splits(response_data, params):
    def pick(cell_line):
        mask = response_data.cell_line_ids == cell_line
        return DrugResponseDataset(
            response=response_data.response[mask],
            cell_line_ids=response_data.cell_line_ids[mask],
            drug_ids=response_data.drug_ids[mask],
            tissues=response_data.tissue[mask],
            dataset_name=response_data.dataset_name,
        )
    return [{
        "train": pick("CL-0"),
        "validation": pick("CL-1"),
        "test": pick("CL-2"),
        "metadata": {
            "random_state": params.random_state,
            "validation_ratio": params.validation_ratio,
            "n_cv_splits": params.n_cv_splits,
            "test_mode": params.test_mode,
            "split_early_stopping": params.split_early_stopping,
        },
    }]
""",
        encoding="utf-8",
    )
    dataset = _sample_dataset(n_cell_lines=3, n_drugs=2)
    splits, metadata = run_custom_splitter(
        dataset,
        script,
        test_mode="LCO",
        n_cv_splits=3,
        validation_ratio=0.2,
        random_state=99,
        split_early_stopping=False,
    )
    assert metadata[0]["random_state"] == 99
    assert metadata[0]["validation_ratio"] == 0.2
    assert metadata[0]["n_cv_splits"] == 3
    assert metadata[0]["test_mode"] == "LCO"
    assert metadata[0]["split_early_stopping"] is False
    assert "validation_es" not in splits[0]
    assert "early_stopping" not in splits[0]


def test_result_discovery_regex_accepts_custom_split_label(tmp_path: Path) -> None:
    """
    Accept arbitrary split-label directories in result discovery regex.

    :param tmp_path: Temporary path provided by pytest.
    """
    import re

    result_dir_str = str(tmp_path).replace("\\", "/")
    pattern = re.compile(
        rf"{result_dir_str}/{re.escape('GDSC1')}/"
        r"[^/]+/[^/]+/(predictions|cross_study|randomization|robustness)/.*\.csv$"
    )
    pred = tmp_path / "GDSC1" / "scaling-lco" / "ElasticNet" / "predictions" / "predictions_split_0.csv"
    pred.parent.mkdir(parents=True, exist_ok=True)
    pred.write_text("cell_line_name,pubchem_id,response,predictions\n", encoding="utf-8")
    assert pattern.match(str(pred).replace("\\", "/"))
