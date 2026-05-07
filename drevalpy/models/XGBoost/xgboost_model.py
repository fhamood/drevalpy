import json
import os

import joblib

import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from drevalpy.models.drp_model import DRPModel
from drevalpy.datasets.dataset import FeatureDataset, DrugResponseDataset

from ..utils import (
    ProteomicsMedianCenterAndImputeTransformer,
    _get_view_as_list,
    load_single_cell_line_view,
    load_single_drug_view,
    scale_gene_expression,
)

class XGBoost(DRPModel):
    """XGBoost model for drug response prediction using gene expression and drug fingerprints."""

    early_stopping = True
    cell_line_views = []
    drug_views = []

    def __init__(self):
        """Initializes the XGBoost

        The model is built in train(). The gene_expression_scalar is set to the StandardScaler() and later fitted
        using the training data only.
        """
        super().__init__()
        self.model = None
        self.gene_expression_scaler = StandardScaler()

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the name of the model.

        :returns: XGBoost
        """
        return "XGBoost"

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features for a single-view xgboost model.

        :param data_path: Path to the data
        :param dataset_name: Name of the dataset
        :returns: FeatureDataset containing the cell line features
        """
        return load_single_cell_line_view(self.cell_line_views, data_path, dataset_name, self.get_model_name())

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset | None:
        """
        Load the drug features for a single-view xgboost model.

        :param data_path: Path to the data
        :param dataset_name: Name of the dataset
        :returns: FeatureDataset containing the drug features
        """
        return load_single_drug_view(self.drug_views, data_path, dataset_name, self.get_model_name())

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        """
        Builds the model, for models that use hyperparameters.

        :param hyperparameters: hyperparameters for the model
        Example:
            self.model = ElasticNet(alpha=hyperparameters["alpha"], l1_ratio=hyperparameters["l1_ratio"])
        """
        self.log_hyperparameters(hyperparameters)
        self.hyperparameters = hyperparameters
        self.cell_line_views = _get_view_as_list(hyperparameters.get("cell_line_views", ["gene_expression"]))
        self.drug_views = _get_view_as_list(hyperparameters.get("drug_views", ["fingerprints"]))

    def train(
            self,
            output: DrugResponseDataset,
            cell_line_input: FeatureDataset,
            drug_input: FeatureDataset | None = None,
            output_earlystopping: DrugResponseDataset | None = None,
            model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """
        Trains the model.

        The number of features is the number of genes + the number of fingerprints.
        :param output: training dataset containing the response output
        :param cell_line_input: training dataset containing gene expression data
        :param drug_input: training dataset containing fingerprints data
        :param output_earlystopping: not needed
        :param model_checkpoint_dir: not needed
        """
        if len(output) > 0:
            if "gene_expression" in self.cell_line_views:
                cell_line_input = scale_gene_expression(
                    cell_line_input=cell_line_input,
                    cell_line_ids=np.unique(output.cell_line_ids),
                    training=True,
                    gene_expression_scaler=self.gene_expression_scaler,
                )
            if len(self.drug_views) == 0:
                # support for single-drug models
                drug_view = None
            else:
                drug_view = self.drug_views[0]

            x = self.get_concatenated_features(
                cell_line_view=self.cell_line_views[0],
                drug_view=drug_view,
                cell_line_ids_output=output.cell_line_ids,
                drug_ids_output=output.drug_ids,
                cell_line_input=cell_line_input,
                drug_input=drug_input,
            )
            self.model.fit(x, output.response)
        else:
            print("No training data provided, will predict NA.")
            self.model = None

    def predict(
            self,
            cell_line_ids: np.ndarray,
            drug_ids: np.ndarray,
            cell_line_input: FeatureDataset,
            drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Predicts the response for the given input.

        :param drug_ids: drug ids
        :param cell_line_ids: cell line ids
        :param drug_input: drug input
        :param cell_line_input: cell line input
        :returns: predicted drug response
        """
        if self.model is None:
            print("No training data was available, predicting NA.")
            return np.array([np.nan] * len(cell_line_ids))

        if "gene_expression" in self.cell_line_views:
            cell_line_input = scale_gene_expression(
                cell_line_input=cell_line_input,
                cell_line_ids=np.unique(cell_line_ids),
                training=False,
                gene_expression_scaler=self.gene_expression_scaler,
            )

        if len(self.drug_views) == 0:
            # support for single-drug models
            drug_view = None
        else:
            drug_view = self.drug_views[0]

        x = self.get_concatenated_features(
            cell_line_view=self.cell_line_views[0],
            drug_view=drug_view,
            cell_line_ids_output=cell_line_ids,
            drug_ids_output=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        return self.model.predict(x)

    def save(self, directory: str) -> None:
        """
        Save the trained model and any associated preprocessing components to the given directory.

        Saves:
        - model.pkl: the trained xgboost model
        - hyperparameters.json: dictionary of model hyperparameters (if present)
        - scaler.pkl: fitted gene expression scaler (if present)

        :param directory: path to the directory where model files will be stored
        :raises ValueError: if the model is not trained
        """
        os.makedirs(directory, exist_ok=True)
        if self.model is None:
            raise ValueError("Cannot save: model is not trained.")

        joblib.dump(self.model, os.path.join(directory, "model.pkl"))
        with open(os.path.join(directory, "hyperparameters.json"), "w") as f:
            json.dump(getattr(self, "hyperparameters", {}), f)
        joblib.dump(self.gene_expression_scaler, os.path.join(directory, "scaler.pkl"))

    @classmethod
    def load(cls, directory: str) -> "XBBoostModel":
        """
        Load a trained XGBoost model and its preprocessing components from disk.

        Loads:
        - model.pkl: the trained sklearn model
        - hyperparameters.json: model hyperparameters
        - scaler.pkl: gene expression scaler (optional)

        :param directory: path to the directory where model files are stored
        :return: an instance of the model with restored state
        :raises FileNotFoundError: if model.pkl is missing
        """
        model_path = os.path.join(directory, "model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} not found")

        instance = cls()

        hyperparams_path = os.path.join(directory, "hyperparameters.json")
        with open(hyperparams_path) as f:
            hyperparameters = json.load(f)
        instance.build_model(hyperparameters)
        instance.model = joblib.load(model_path)

        scaler_path = os.path.join(directory, "scaler.pkl")
        if os.path.exists(scaler_path):
            instance.gene_expression_scaler = joblib.load(scaler_path)

        return instance

