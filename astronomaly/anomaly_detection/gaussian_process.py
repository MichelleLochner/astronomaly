import pickle
from os import path
import logging

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Matern

from astronomaly.base.base_pipeline import PipelineStage


class GaussianProcess(PipelineStage):

    def __init__(self, **kwargs):
        """
        Runs sklearn's isolation forest anomaly detection algorithm and returns
        the anomaly score for each instance.

        Parameters
        ----------
        contamination : string or float, optional
            Hyperparameter to pass to IsolationForest. 'auto' is recommended

        """
        logging.debug('GaussianProcess passing on kwargs {}'.format(kwargs))
        super().__init__(**kwargs)

        self.estimator = None


    def save_estimator(self):
        """
        Stores the estimator object to the output directory to allow quick 
        rerunning on new data.
        """
        if self.estimator is not None:
            f = open(path.join(self.output_dir, 'estimator.pickle'), 'wb')
            pickle.dump(self.estimator, f)

    def update(self, data):
        feature_cols = [col for col in data.columns.values if col not in ['human_label', 'score', 'trained_score', 'acquisition']]
        features = data[feature_cols]
        labels = data['human_label']

        kernel = RBF() + WhiteKernel()
        self.estimator = GaussianProcessRegressor(kernel=kernel)

        is_labelled = ~pd.isnull(labels)
        y_labelled = labels[is_labelled]  
        X_labelled = features[is_labelled]

        print(y_labelled)
        if np.any(y_labelled < -0.01):
            raise ValueError('Some human labels are negative in {}'.format(y_labelled))

        self.estimator.fit(X_labelled, y_labelled)



        scores, std = self.estimator.predict(features, return_std=True)

        if self.save_output:  # from PipelineStage superclass
            self.save_estimator()

        return pd.DataFrame(
            index=features.index,
            data=np.stack([scores, scores, std], axis=1),  # duplicate score as trained_score so astronomaly understands it is also post-human
            columns=['score', 'trained_score', 'acquisition']
        )


    def _execute_function(self, data):
        """
        Does the work in actually running estimator.

        Parameters
        ----------
        data : pd.DataFrame or similar
            The input features to run estimator on. Assumes the index is the id 
            of each object and all columns are to
            be used as features.

        Returns
        -------
        pd.DataFrame
            Contains the same original index of the features input and the 
            anomaly scores. More negative is more anomalous.

        """

        gp_output = self.update(data)
        # index will def match data...right?
        return gp_output