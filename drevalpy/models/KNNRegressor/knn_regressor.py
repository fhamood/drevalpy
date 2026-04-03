from typing import Any
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

from drevalpy.models.drp_model import DRPModel
from drevalpy.datasets.dataset import FeatureDataset, DrugResponseDataset
from drevalpy.models.utils import (
    load_and_select_gene_features,
    load_drug_fingerprint_features,
)

class KNNRegressor(DRPModel):
    """A revolutionary new modeling strategy."""

    is_single_drug_model = False #false, because Drug-Features like fingerprints are given
    early_stopping = False #false, because by default KNNRepressor does one run
    cell_line_views = ["gene_expression"]
    drug_views = ["fingerprints"]

    def __init__(self):
        super().__init__()

        self.model: KNeighborsRegressor 
        self.hyperparameters: dict[str, Any] = {}
        self.scaler_gex = StandardScaler()
        self.pca = None
        self.featureshape = None

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the name of the model.
        """
        return "KNNRegressor"
    
    @classmethod
    def load_cell_line_features(cls, data_path: str, dataset_name: str) -> FeatureDataset:
        """Loads the cell line features.

        :param data_path: Path to the gene expression and landmark genes
        :param dataset_name: name of the dataset
        :return: FeatureDataset containing the cell line gene expression features.
        """
        return load_and_select_gene_features(
            feature_type="gene_expression",
            gene_list="None",
            data_path=data_path,
            dataset_name=dataset_name,
        )
    
    @classmethod
    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:

        return load_drug_fingerprint_features(data_path, dataset_name, fill_na=True)
    
    def build_model(self, hyperparameters: dict[str, Any]) -> None:
         """
        Builds the model, for models that use hyperparameters.

        :param hyperparameters: hyperparameters for the model
        Example:
            self.model = ElasticNet(alpha=hyperparameters["alpha"], l1_ratio=hyperparameters["l1_ratio"])
        """
         #self.log_hyperparameters(hyperparameters)
         self.hyperparameters = hyperparameters

         n_neighbors = hyperparameters.get("n_neighbors")
         weights = hyperparameters.get("weights", "distance")
         
         self.model = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights = weights,
        )
    
    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:
        
        if drug_input is None:
            raise ValueError("drug_input (fingerprints) is required.")
        
        x = self.get_concatenated_features(
            cell_line_view=self.cell_line_views[0],
            drug_view=self.drug_views[0],
            cell_line_ids_output=output.cell_line_ids,
            drug_ids_output=output.drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        self.featureshape = x.shape[1]
        x = normalize(X=x)
        self.pca = PCA(n_components= self.hyperparameters.get("variance", 0.75))
        x = self.pca.fit_transform(X = x)

        self.model.fit(x, output.response)
        
    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        
        if drug_input is None:
            raise ValueError("drug_input (fingerprints) is required.")
        
        if self.model is None:
            return np.full(len(cell_line_ids), np.nan)
        
        x = self.get_concatenated_features(
            cell_line_view=self.cell_line_views[0],
            drug_view=self.drug_views[0],
            cell_line_ids_output=cell_line_ids,
            drug_ids_output=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        x = normalize(X = x)
        if x.shape[1] != self.featureshape:
            raise ValueError("input feature size is unequall to training feature size")
        x = self.pca.transform(X=x)
        return self.model.predict(x)