"""Tests for the Lasso Model"""

import tempfile

import numpy as np
import pytest

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.evaluation import evaluate
from drevalpy.experiment import cross_study_prediction
from drevalpy.models import MODEL_FACTORY

from drevalpy.models.drp_model import DRPModel


def test_lasso_get_model_name() -> None:
    """
    Test that the model name is returned correctly.
    """
    model = MODEL_FACTORY["Lasso"]()
    assert model.get_model_name() == "Lasso"


def test_lasso_build_model() -> None:
    """
    Test that build_model stores hyperparameters and creates a sklearn Lasso model.
    """
    model = MODEL_FACTORY["Lasso"]()
    model.build_model({"alpha": 0.001})

    assert model.hyperparameters == {"alpha": 0.001}
    assert model.model is not None
    assert model.model.alpha == 0.001


def test_lasso_class_attributes() -> None:
    """
    Test that class attributes are set correctly.
    """
    model = MODEL_FACTORY["Lasso"]()

    assert model.early_stopping is False
    assert model.is_single_drug_model is False
    assert model.cell_line_views == ["gene_expression"]
    assert model.drug_views == ["fingerprints"]


def _call_lasso(
    train_dataset: DrugResponseDataset,
    val_dataset: DrugResponseDataset,
    model: DRPModel,
    cell_line_input: FeatureDataset,
    drug_input: FeatureDataset,
    test_mode: str,
    alpha: float
) -> DRPModel:
    """
    Train and validate the Lasso model.

    :param train_dataset: training dataset
    :param val_dataset: validation dataset
    :param model: Lasso model instance
    :param cell_line_input: cell line features
    :param drug_input: drug features
    :param test_mode: either LPO, LCO, LDO, or LTO
    :param alpha: regularization strength 
    :returns: trained Lasso model
    """
    model.build_model({"alpha": alpha})

    model.train(
        output=train_dataset,
        cell_line_input=cell_line_input,
        drug_input=drug_input,
    )

    val_dataset._predictions = model.predict(
        drug_ids=val_dataset.drug_ids,
        cell_line_ids=val_dataset.cell_line_ids,
        drug_input=drug_input,
        cell_line_input=cell_line_input,
    )

    assert val_dataset.predictions is not None
    assert isinstance(val_dataset.predictions, np.ndarray)
    assert val_dataset.predictions.shape == (len(val_dataset),)
    assert np.all(np.isfinite(val_dataset.predictions))

    metrics = evaluate(val_dataset, metric=["Pearson"])
    assert metrics["Pearson"] >= -1
    
    print(f"{test_mode}: Lasso prediction successful on validation split")

    return model


@pytest.mark.parametrize("test_mode", ["LTO", "LPO", "LCO", "LDO"])
@pytest.mark.parametrize("alpha", [0.0001, 0.001])
def test_lasso(
    sample_dataset: DrugResponseDataset,
    test_mode: str,
    cross_study_dataset: DrugResponseDataset,
    alpha: float,
) -> None:
    """
    Test the Lasso model.

    :param sample_dataset: from conftest.py
    :param test_mode: either LPO, LCO, LDO, or LTO
    :param cross_study_dataset: dataset
    :raises ValueError: if drug input is None
    """
    drug_response = sample_dataset
    drug_response.split_dataset(
        n_cv_splits=2,
        mode=test_mode,
        validation_ratio=0.4,
    )
    assert drug_response.cv_splits is not None
    
    split = drug_response.cv_splits[0]
    train_dataset = split["train"]
    val_dataset = split["validation"]

    model = MODEL_FACTORY["Lasso"]()
    cell_line_input = model.load_cell_line_features(data_path="../data", dataset_name="TOYv1")
    drug_input = model.load_drug_features(data_path="../data", dataset_name="TOYv1")

    if drug_input is None:
        raise ValueError("Drug input is None")

    cell_lines_to_keep = cell_line_input.identifiers
    drugs_to_keep = drug_input.identifiers

    len_train_before = len(train_dataset)
    len_pred_before = len(val_dataset)
    
    train_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    val_dataset.reduce_to(cell_line_ids=cell_lines_to_keep, drug_ids=drugs_to_keep)
    print(f"Reduced training dataset from {len_train_before} to {len(train_dataset)}")
    print(f"Reduced val dataset from {len_pred_before} to {len(val_dataset)}")

    model = _call_lasso(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        cell_line_input=cell_line_input,
        drug_input=drug_input,
        test_mode=test_mode,
        alpha=alpha
    )

    # Save and load test
    with tempfile.TemporaryDirectory() as model_dir:
        model.save(model_dir)
        loaded_model = MODEL_FACTORY["Lasso"].load(model_dir)

        preds_before = model.predict(
            drug_ids=val_dataset.drug_ids,
            cell_line_ids=val_dataset.cell_line_ids,
            drug_input=drug_input,
            cell_line_input=cell_line_input,
        )
        preds_after = loaded_model.predict(
            drug_ids=val_dataset.drug_ids,
            cell_line_ids=val_dataset.cell_line_ids,
            drug_input=drug_input,
            cell_line_input=cell_line_input,
        )
        assert isinstance(preds_before, np.ndarray)
        assert isinstance(preds_after, np.ndarray)
        assert preds_after.shape == preds_before.shape
        assert np.all(np.isfinite(preds_before))
        assert np.all(np.isfinite(preds_after))

    # make temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Running cross-study prediction for Lasso")
        cross_study_prediction(
            dataset=cross_study_dataset,
            model=model,
            test_mode=test_mode,
            train_dataset=train_dataset,
            path_data="../data",
            early_stopping_dataset=None,
            response_transformation=None,
            path_out=temp_dir,
            split_index=0,
            single_drug_id=None,
        )


def test_lasso_train_requires_drug(sample_dataset: DrugResponseDataset) -> None:
    """
    Test that Lasso raises an error if drug_input is missing during train.

    :param sample_dataset: sample dataset from conftest.py
    :raises ValueError: if drug input is None
    """
    drug_response = sample_dataset
    drug_response.split_dataset(
        n_cv_splits=2,
        mode="LPO",
        validation_ratio=0.4,
    )
    assert drug_response.cv_splits is not None

    split = drug_response.cv_splits[0]
    train_dataset = split["train"]

    model = MODEL_FACTORY["Lasso"]()
    model.build_model({"alpha": 0.001})

    with pytest.raises(ValueError, match="drug_input"):
        model.train(
            output=train_dataset,
            cell_line_input=None,
            drug_input=None,
        )


def test_lasso_predict_requires_drug(sample_dataset: DrugResponseDataset) -> None:
    """
    Test that Lasso raises an error if drug_input is missing during predict.

    :param sample_dataset: sample dataset from conftest.py
    :raises ValueError: if drug input is None
    """
    drug_response = sample_dataset
    drug_response.split_dataset(
        n_cv_splits=2,
        mode="LPO",
        validation_ratio=0.4,
    )
    assert drug_response.cv_splits is not None

    split = drug_response.cv_splits[0]
    val_dataset = split["validation"]

    model = MODEL_FACTORY["Lasso"]()

    with pytest.raises(ValueError, match="drug_input"):
        model.predict(
            drug_ids=val_dataset.drug_ids,
            cell_line_ids=val_dataset.cell_line_ids,
            drug_input=None,
            cell_line_input=None, 
        )


def test_lasso_predict_nan_untrained(sample_dataset: DrugResponseDataset) -> None:
    """
    Test that Lasso returns NaN predictions if the model has not been built.

    :param sample_dataset: sample dataset from conftest.py
    :raises ValueError: if drug input is None
    """
    drug_response = sample_dataset
    drug_response.split_dataset(
        n_cv_splits=2,
        mode="LPO",
        validation_ratio=0.4,
    )
    assert drug_response.cv_splits is not None

    split = drug_response.cv_splits[0]
    val_dataset = split["validation"]

    model = MODEL_FACTORY["Lasso"]()
    cell_line_input = model.load_cell_line_features(data_path="data", dataset_name="TOYv1")
    drug_input = model.load_drug_features(data_path="data", dataset_name="TOYv1")

    if drug_input is None:
        raise ValueError("Drug input is None")

    val_dataset.reduce_to(
        cell_line_ids=cell_line_input.identifiers,
        drug_ids=drug_input.identifiers,
    )

    preds = model.predict(
        drug_ids=val_dataset.drug_ids,
        cell_line_ids=val_dataset.cell_line_ids,
        drug_input=drug_input,
        cell_line_input=cell_line_input,
    )

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(val_dataset),)
    assert np.all(np.isnan(preds))


def test_lasso_train_without_build(sample_dataset: DrugResponseDataset) -> None:
    """
    Test that Lasso raises an error if train is called before build_model.
    
    :param sample_dataset: sample dataset from conftest.py
    :raises ValueError: if model was not built before training
    """
    drug_response = sample_dataset
    drug_response.split_dataset(
        n_cv_splits=2,
        mode="LPO",
        validation_ratio=0.4,
    )
    assert drug_response.cv_splits is not None

    split = drug_response.cv_splits[0]
    train_dataset = split["train"]

    model = MODEL_FACTORY["Lasso"]()
    cell_line_input = model.load_cell_line_features(data_path="../data", dataset_name="TOYv1")
    drug_input = model.load_drug_features(data_path="../data", dataset_name="TOYv1")

    if drug_input is None:
        raise ValueError("Drug input is None")

    train_dataset.reduce_to(
        cell_line_ids=cell_line_input.identifiers,
        drug_ids=drug_input.identifiers,
    )

    with pytest.raises(ValueError, match="Model has not been built yet"):
        model.train(
            output=train_dataset,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )