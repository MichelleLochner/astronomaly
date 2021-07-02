from astronomaly.base.base_pipeline import PipelineStage
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
import pickle
from os import path


class LOF_Algorithm(PipelineStage):
    def __init__(self, contamination='auto', n_neighbors=20, **kwargs):
        """
        Runs sklearn's local outlier factor anomaly detection algorithm and 
        returns the anomaly score for each instance.

        Parameters
        ----------
        contamination : string or float, optional
            Hyperparameter to pass to LOF. 'auto' is recommended
        n_neighbors : int
            Hyperparameter to pass to LOF. Fairly sensitive to the amount of 
            data in the dataset.

        """
        super().__init__(
            contamination=contamination, n_neighbors=n_neighbors, **kwargs)

        self.contamination = contamination
        self.n_neighbors = n_neighbors

        self.algorithm_obj = None

    def save_algorithm_obj(self):
        """
        Stores the LOF object to the output directory to allow quick 
        rerunning on new data.
        """
        if self.algorithm_obj is not None:
            f = open(path.join(self.output_dir, 
                               'ml_algorithm_object.pickle'), 'wb')
            pickle.dump(self.algorithm_obj, f)

    def _execute_function(self, features):
        """
        Does the work in actually running the algorithm.

        Parameters
        ----------
        features : pd.DataFrame or similar
            The input features to run the algorithm on. Assumes the index is 
            the id of each object and all columns are to be used as features.

        Returns
        -------
        pd.DataFrame
            Contains the same original index of the features input and the 
            anomaly scores. More negative is more anomalous.

        """
        self.algorithm_obj = LocalOutlierFactor(
            contamination=self.contamination, 
            n_neighbors=self.n_neighbors, 
            novelty=False)

        # why fit_predict and not fit? does it trigger negative_outlier_factor_?
        self.algorithm_obj.fit_predict(features)

        scores = self.algorithm_obj.negative_outlier_factor_

        if self.save_output:
            self.save_algorithm_obj()

        return pd.DataFrame(data=scores, index=features.index, 
                            columns=['score'])
