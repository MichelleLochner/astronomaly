from sklearn.preprocessing import StandardScaler
import pandas as pd
from astronomaly.base.base_pipeline import PipelineStage


class FeatureScaler(PipelineStage):
    def __init__(self, **kwargs):
        """
        Rescales features using a standard sklearn scalar that subtracts the 
        mean and divides by the standard deviation for each feature. Highly 
        recommended for most machine learning algorithms and for any data 
        visualisation such as t-SNE.
        """
        super().__init__(**kwargs)

    def _execute_function(self, features):
        """
        Does the work in actually running the scaler.

        Parameters
        ----------
        features : pd.DataFrame or similar
            The input features to run iforest on. Assumes the index is the id 
            of each object and all columns are to be used as features.

        Returns
        -------
        pd.DataFrame
            Contains the same original index and columns of the features input 
            with the features scaled to zero mean and unit variance.

        """
        scl = StandardScaler()
        output = scl.fit_transform(features)

        return pd.DataFrame(data=output, index=features.index, 
                            columns=features.columns)
