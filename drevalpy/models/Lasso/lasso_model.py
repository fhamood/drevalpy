# Lasso = Least Absolute Shrinkage and Selection Operator
# > linear regression model with L1 regularization


from typing import Any

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

from drevalpy.datasets.dataset import DrugResponseDataset, FeatureDataset
from drevalpy.models.drp_model import DRPModel
from drevalpy.models.utils import (
    load_and_select_gene_features,
    load_drug_fingerprint_features,
)


class LassoModel(DRPModel):
    """
    Lasso regression model for drug response prediction.

    This model combines cell line gene expression features and drug fingerprint
    features into a single input matrix and uses a sklearn Lasso model to predict
    drug response values.

    Lasso applies L1 regularization, which shrinks coefficients to zero.
    Some coefficients can become zero, enabling feature selection and reducing model complexity.
    """

    # Used in the pipeline!
    early_stopping = False
    is_single_drug_model = False

    cell_line_views = ["gene_expression"]
    drug_views = ["fingerprints"]

    def __init__(self):
        """
        Initializes the Lasso model.

        self.model: stores the sklearn Lasso model
        self.input_scaler: scales the combined input matrix
        self.hyperparameters: stores the passed hyperparameters
        """
        super().__init__()
        self.model = None
        self.input_scaler = StandardScaler()
        self.hyperparameters = None

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.
        """
        return "Lasso"

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads gene expression features for cell lines.

        :param data_path: path to the data directory
        :param dataset_name: name of the dataset
        :returns: FeatureDataset containing gene expression features
        """
        return load_and_select_gene_features(
            feature_type="gene_expression",
            data_path=data_path,
            dataset_name=dataset_name,
            gene_list="landmark_genes",
        )

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Load drug features from fingerprints.

        :param data_path: path to the data directory
        :param dataset_name: name of the dataset
        :returns: FeatureDataset containing drug fingerprint features
        """
        return load_drug_fingerprint_features(data_path=data_path, dataset_name=dataset_name, fill_na=True)

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        """
        Build the sklearn Lasso model using hyperparameters.

        The hyperparameters are stored in the model instance and then passed to "sklearn.linear_model.Lasso".

        :param hyperparameters: dictionary containing model hyperparameters
        """
        self.hyperparameters = hyperparameters

        self.model = Lasso(
            alpha=hyperparameters["alpha"],
            max_iter=10000,  
            tol=1e-3,
            selection="random",
        )

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        """
        Train the Lasso model on gene expression and drug fingerprint features.

        Procedure:
        - get gene expression data for all the cell lines
        - get fingerprints for the drugs
        - combine both feature matrices
        - scale the combined input matrix
        - get target values
        - Lasso fit

        :param output: training dataset containing response values, cell line ids, and drug ids
        :param cell_line_input: FeatureDataset containing cell line features
        :param drug_input: FeatureDataset containing drug features
        :raises ValueError: if drug_input is None
        """
        if drug_input is None:
            raise ValueError("drug_input (fingerprints) is required for LassoModel.")

        # Retrieve gene expression features for the training cell lines
        gex = cell_line_input.get_feature_matrix("gene_expression", output.cell_line_ids)

        # Retrieve fingerprint features for the corresponding drugs
        fp = drug_input.get_feature_matrix("fingerprints", output.drug_ids)

        # Concatenate cell line and drug features column-wise
        x = np.concatenate([gex, fp], axis=1)
        x = self.input_scaler.fit_transform(x)

        # Target vector containing the drug response values
        y = output.response

        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model first.")

        # Fit the Lasso model
        self.model.fit(x, y)

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        """
        Predict drug response values.

        Procedure:
        - load appropriate cell lines festures and drug features
        - combine features
        - transform the combined feature matrix with the training scaler
        - generate predictions with the trained model

        :param cell_line_ids: array of cell line identifiers
        :param drug_ids: array of drug identifiers
        :param cell_line_input: FeatureDataset containing cell line features
        :param drug_input: FeatureDataset containing drug features
        :returns: predicted drug response values
        :raises ValueError: if drug_input is None

        """
        if drug_input is None:
            raise ValueError("drug_input (fingerprints) is required for LassoModel.")

        # If the model has not been trained yet, return NaN predictions
        if self.model is None:
            return np.full(len(cell_line_ids), np.nan)

        # Retrieve gene expression and fingerprint features
        gex = cell_line_input.get_feature_matrix("gene_expression", cell_line_ids)

        fp = drug_input.get_feature_matrix("fingerprints", drug_ids)

        # Concatenate cell line and drug features
        x = np.concatenate([gex, fp], axis=1)
        x = self.input_scaler.transform(x)

        # Predict drug response values
        return self.model.predict(x)