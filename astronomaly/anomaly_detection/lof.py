from astronomaly.base.base_pipeline import PipelineStage
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
import pickle
from os import path


class LOF_Algorithm(PipelineStage):
    def __init__(self, contamination='auto', **kwargs):
        """
        Runs sklearn's isolation forest anomaly detection algorithm and returns the anomaly score for each instance.

        Parameters
        ----------
        contamination : string or float, optional
            Hyperparameter to pass to IsolationForest. 'auto' is recommended

        """
        super().__init__(contamination=contamination, **kwargs)

        self.contamination = contamination

        self.algorithm_obj = None

    def save_algorithm_obj(self):
        """
        Stores the iforest object to the output directory to allow quick rerunning on new data.
        """
        if self.algorithm_obj is not None:
            f = open(path.join(self.output_dir, 'ml_algorithm_object.pickle'), 'wb')
            pickle.dump(self.algorithm_obj, f)

    def _execute_function(self, features):
        """
        Does the work in actually running isolation forest.

        Parameters
        ----------
        features : pd.DataFrame or similar
            The input features to run iforest on. Assumes the index is the id of each object and all columns are to
            be used as features.

        Returns
        -------
        pd.DataFrame
            Contains the same original index of the features input and the anomaly scores. More negative is
            more anomalous.

        """
        self.algorithm_obj = LocalOutlierFactor(contamination=self.contamination, novelty=True)
        self.algorithm_obj.fit(features)

        scores = self.algorithm_obj.decision_function(features)

        if self.save_output:
            self.save_algorithm_obj()

        return pd.DataFrame(data=scores, index=features.index, columns=['score'])
