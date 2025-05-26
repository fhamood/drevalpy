from drevalpy.models.drp_model import DRPModel
from drevalpy.datasets.dataset import FeatureDataset

from ..utils import load_and_select_gene_features, load_drug_fingerprint_features

class MultiProteomicsRandomForest(DRPModel):
    """RandomForest model for drug response prediction using proteomics data. Can use proteomics,
    phosphoproteomics, and kinase scores as input features. Requires kinobeads as drug features."""

    is_single_drug_model = False
    early_stopping = False
    cell_line_views = ["proteomics", "phosphoproteomics", "kinase_scores"]
    drug_views = ["kinobeads"]

    def get_model_name(cls) -> str:
        """
        Returns the name of the model.
        """
        return "MultiProteomicsRandomForest"

    def load_cell_line_features(self, data_path: str, dataset_name: str) -> FeatureDataset:
        """
        Loads the cell line features.

        :param data_path: Path to the gene expression and landmark genes
        :param dataset_name: Name of the dataset
        :returns: FeatureDataset containing the cell line proteomics features, filtered through the landmark genes
        """
        return load_and_select_gene_features(
            feature_type="proteomics",
            gene_list=None,
            data_path=data_path,
            dataset_name=dataset_name,
        )

    def load_drug_features(self, data_path: str, dataset_name: str) -> FeatureDataset | None:
        """
        Load the drug features, in this case the fingerprints.

        :param data_path: Path to the data
        :param dataset_name: Name of the dataset
        :returns: FeatureDataset containing the drug fingerprints
        """
        return load_drug_fingerprint_features(data_path, dataset_name, fill_na=True)

