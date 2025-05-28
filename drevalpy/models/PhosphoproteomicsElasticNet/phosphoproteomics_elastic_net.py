from drevalpy.models.baselines.sklearn_models import ElasticNetModel, ProteomicsElasticNetModel
from drevalpy.datasets.dataset import FeatureDataset, DrugResponseDataset
import numpy as np


class PhosphoproteomicsElasticNetModel(ElasticNetModel):

    cell_line_views = ["phosphoproteomics"]
    drug_views = ["onehot"]

    def __init__(self):
        super().__init__()

    @classmethod
    def get_model_name(cls) -> str:
        return "PhosphoproteomicsElasticNet"

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        feature_dataset = FeatureDataset.from_csv(f"{data_path}/{dataset_name}/{self.cell_line_views[0]}.csv",
                                                  id_column='cell_line_name', view_name=self.cell_line_views[0])

        return feature_dataset

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        feature_dataset = FeatureDataset.from_csv(f"{data_path}/{dataset_name}/drug_features_{self.drug_views[0]}.csv",
                                                  id_column='pubchem_id', view_name=self.drug_views[0])

        return feature_dataset


    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:

        if drug_input is None:
            raise ValueError(f"drug_input ({self.drug_views[0]}) is required for the sklearn models.")

        x = self.get_concatenated_features(
            cell_line_view=self.cell_line_views[0],
            drug_view=self.drug_views[0],
            cell_line_ids_output=output.cell_line_ids,
            drug_ids_output=output.drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        self.model.fit(x, output.response)

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:
        if drug_input is None:
            raise ValueError(f"drug_input ({self.drug_views[0]}) is required.")
        if self.model is None:
            print("No training data was available, or model not trained predicting NA.")
            return np.array([np.nan] * len(cell_line_ids))
        x = self.get_concatenated_features(
            cell_line_view=self.cell_line_views[0],
            drug_view=self.drug_views[0],
            cell_line_ids_output=cell_line_ids,
            drug_ids_output=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=drug_input,
        )
        return self.model.predict(x)
