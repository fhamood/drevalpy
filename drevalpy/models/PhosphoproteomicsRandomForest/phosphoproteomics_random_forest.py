from drevalpy.models.drp_model import DRPModel
from drevalpy.datasets.dataset import FeatureDataset
from drevalpy.models import RandomForest

import pandas as pd

class PhosphoproteomicsRandomForest(RandomForest):
    """RandomForest model for drug response prediction using proteomics data."""

    cell_line_views = ["proteomics"]

    @classmethod
    def get_model_name(cls) -> str:
        """
        Returns the model name.

        :returns: ProteomicsRandomForest
        """
        return "ProteomicsRandomForest"
