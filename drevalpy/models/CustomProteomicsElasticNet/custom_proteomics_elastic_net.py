from drevalpy.models.baselines.sklearn_models import ElasticNetModel, ProteomicsElasticNetModel
from drevalpy.datasets.dataset import FeatureDataset, DrugResponseDataset
import numpy as np


class CustomProteomicsElasticNetModel(ElasticNetModel):
    """ElasticNet model for drug response prediction using proteomics data."""

    cell_line_views = ["proteomics"]
    drug_views = ["onehot"]

    def __init__(self):
        """
        Initializes the model with specific hyperparameters.

        feature_threshold: for feature selection. Require that, e.g., 70% of the proteins are measured without NAs
        over all cell lines -> n_complete_features = number of proteins with at least 70% of the cell lines
        n_features: fallback for feature selection. Take top n complete features.
        Select max(n_complete_features, n_features) features.
        normalization_width: width of the Gaussian kernel for the median centering
        normalization_downshift: downshift of the median for the imputation of missing values
        """
        super().__init__()
        self.feature_threshold = 0.7
        self.n_features = 1000
        self.normalization_width = 0.3
        self.normalization_downshift = 1.8

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: ProteomicsElasticNet
        """
        return "CustomProteomicsElasticNet"

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features.

        :param data_path: Path to the gene expression and landmark genes
        :param dataset_name: Name of the dataset
        :returns: FeatureDataset containing the cell line proteomics features, filtered through the landmark genes
        """
        feature_dataset = FeatureDataset.from_csv(f"{data_path}/{dataset_name}/proteomics.csv",
                                                  id_column='cell_line_name', view_name="proteomics",
                                                  drop_columns=['cellosaurus_id'])

        return feature_dataset

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the drug features, in this case the drug ids.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: FeatureDataset containing the drug ids
        """
        feature_dataset = FeatureDataset.from_csv(f"{data_path}/{dataset_name}/drug_features_onehot.csv",
                                                  id_column='pubchem_id', view_name="onehot")

        return feature_dataset


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
        :raises ValueError: If drug_input is None.
        """
        if drug_input is None:
            raise ValueError("drug_input (fingerprints) is required for the sklearn models.")

        # # Not needed because we already prepared the data
        # cell_line_input = prepare_proteomics_custom(
        #     cell_line_input=cell_line_input,
        #     cell_line_ids=np.unique(output.cell_line_ids),
        #     training=True,
        #     transformer=self.proteomics_transformer,
        # )
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
        """
        Predicts the response for the given input.

        :param drug_ids: drug ids
        :param cell_line_ids: cell line ids
        :param drug_input: drug input
        :param cell_line_input: cell line input
        :returns: predicted drug response
        :raises ValueError: If drug_input is None.
        """
        if drug_input is None:
            raise ValueError("drug_input (fingerprints) is required.")
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
