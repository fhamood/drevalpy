"""SingleDrugElasticNet and SingleDrugRandomForest class. Fit a model for each drug separately."""

from .sklearn_models import ElasticNetModel, RandomForest


class SingleDrugElasticNet(ElasticNetModel):
    """SingleDrugElasticNet class."""

    is_single_drug_model = True
    drug_views = []
    early_stopping = False

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: SingleDrugElasticNet
        """
        return "SingleDrugElasticNet"

    def load_drug_features(self, data_path, dataset_name):
        """
        Load drug features. Not needed for SingleDrugElasticNet.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: None
        """
        return None


class SingleDrugRandomForest(RandomForest):
    """SingleDrugRandomForest class."""

    is_single_drug_model = True
    drug_views = []
    early_stopping = False

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: SingleDrugRandomForest
        """
        return "SingleDrugRandomForest"

    def load_drug_features(self, data_path, dataset_name):
        """
        Load drug features. Not needed for SingleDrugRandomForest.

        :param data_path: path to the data
        :param dataset_name: name of the dataset
        :returns: None
        """
        return None
