from drevalpy.models.drp_model import DRPModel
from drevalpy.models.baselines.sklearn_models import ElasticNet
from drevalpy.datasets.dataset import FeatureDataset

import pandas as pd

class CustomSingleDrugElasticNet(ElasticNet):
    """Let's see if Manuel's model works"""

    is_single_drug_model = True
    early_stopping = False
    cell_line_views = ["proteomics", "phosphoproteomics"]

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the name of the model.
        """
        return "SingleDrugElasticNet"


    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features, in this case the cell line ids.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: FeatureDataset containing the cell line ids
        """
        feature_dataset = FeatureDataset.from_csv(f"{data_path}/{dataset_name}_gene_expression.csv",
                                                  id_column="cell_line_ids",
                                                  view_name="gene_expression"
                                                  )  # make sure to adjust the path to your data
        phosphoproteome = FeatureDataset.from_csv(f"{data_path}/{dataset_name}_methylation.csv",
                                              id_column="cell_line_ids",
                                              view_name="gene_expression"
                                              )  # make sure to adjust the path to your data
        feature_dataset.add_features(phosphoproteome)

        return feature_dataset

    def build_model(self, hyperparameters: dict[str, Any]) -> None:
        """
        Builds the model for models that use hyperparameters.

        :param hyperparameters: hyperparameters for the model
        Example:
            self.model = ElasticNet(alpha=hyperparameters["alpha"], l1_ratio=hyperparameters["l1_ratio"])
        """
        self.model = ElasticNet(alpha=hyperparameters["alpha"], l1_ratio=hyperparameters["l1_ratio"])