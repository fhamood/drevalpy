from drevalpy.models.baselines.singledrug_elastic_net import SingleDrugElasticNet
from drevalpy.datasets.dataset import FeatureDataset, DrugResponseDataset
import numpy as np


class CustomSingleDrugElasticNet(SingleDrugElasticNet):
    def __init__(self):
        super().__init__()

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        feature_dataset = FeatureDataset.from_csv(f"{data_path}/{dataset_name}/{self.cell_line_views[0]}.csv",
                                                  id_column='cell_line_name', view_name=f"{self.cell_line_views[0]}",
                                                  drop_columns=['cellosaurus_id'])
        return feature_dataset

    def train(
        self,
        output: DrugResponseDataset,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
        output_earlystopping: DrugResponseDataset | None = None,
        model_checkpoint_dir: str = "checkpoints",
    ) -> None:

        x = self.get_concatenated_features(
            cell_line_view=self.cell_line_views[0],
            drug_view=None,
            cell_line_ids_output=output.cell_line_ids,
            drug_ids_output=output.drug_ids,
            cell_line_input=cell_line_input,
            drug_input=None,
        )
        self.model.fit(x, output.response)

    def predict(
        self,
        cell_line_ids: np.ndarray,
        drug_ids: np.ndarray,
        cell_line_input: FeatureDataset,
        drug_input: FeatureDataset | None = None,
    ) -> np.ndarray:

        if self.model is None:
            print("No training data was available, or model not trained predicting NA.")

            return np.array([np.nan] * len(cell_line_ids))
        x = self.get_concatenated_features(
            cell_line_view=self.cell_line_views[0],
            drug_view=None,
            cell_line_ids_output=cell_line_ids,
            drug_ids_output=drug_ids,
            cell_line_input=cell_line_input,
            drug_input=None,
        )
        return self.model.predict(x)



class CustomSingleDrugProteomicsElasticNet(CustomSingleDrugElasticNet):

    cell_line_views = ["proteomics"]

    def __init__(self):
        super().__init__()

    @classmethod
    def get_model_name(cls) -> str:
        return "CustomSingleDrugProteomicsElasticNet"


class SingleDrugPhosphoproteomicsElasticNet(CustomSingleDrugElasticNet):

    cell_line_views = ["phosphoproteomics"]

    def __init__(self):
        super().__init__()

    @classmethod
    def get_model_name(cls) -> str:
        return "SingleDrugPhosphoproteomicsElasticNet"


class SingleDrugKinaseScoreElasticNet(CustomSingleDrugElasticNet):

    cell_line_views = ["kinase_scores"]

    def __init__(self):
        super().__init__()

    @classmethod
    def get_model_name(cls) -> str:
        return "SingleDrugKinaseScoreElasticNet"