import pickle
from os import path
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from astronomaly.base.base_pipeline import PipelineStage


class GaussianProcess(PipelineStage):

    def __init__(self, features, ei_tradeoff=0.5, **kwargs):
        """
        Runs a Gaussian process regression algorithm to automatically find 
        regions of interest without first running an anomaly detection 
        algorithm.

        Parameters
        ----------
        features : pd.DataFrame
            The features on which the Gaussian process will be run. The human 
            labels can only be provided at runtime, but the features are not 
            expected to change.
        ei_tradeoff : float, optional
            The expected improvement trade off. Balances exploration against
            exploitation. Default is 0.5. 

        """
        super().__init__(features=features, ei_tradeoff=ei_tradeoff, **kwargs)

        self.estimator = None
        self.features = features
        self.ei_tradeoff = ei_tradeoff

    def save_estimator(self):
        """
        Stores the estimator object to the output directory to allow quick 
        rerunning on new data.
        """
        if self.estimator is not None:
            f = open(path.join(self.output_dir, 'gp_estimator.pickle'), 'wb')
            pickle.dump(self.estimator, f)

    def EI(self, mean, std, max_val):
        """
        Calculates the expected improvement to use as an acquisition function.
        Same as used in Walmsley et al. 2021. Code taken from 
        https://modal-python.readthedocs.io/en/latest/_modules/modAL/acquisition.html#max_EI

        Parameters
        ----------
        mean : np.array
            The predicted mean from the GP
        std : np.array
            The standard deviation from the GP
        max_val : float
            The current maximum user score seen so far
        """
        tradeoff = self.ei_tradeoff
        z = (mean - max_val - tradeoff) / std
        return (mean - max_val - tradeoff) * ndtr(z) + std * norm.pdf(z)

    def update(self, data):
        feature_cols = [col for col in data.columns.values if col not in
                        ['human_label', 
                         'score', 
                         'trained_score', 
                         'acquisition']]
        features = data[feature_cols]

        if 'human_label' in data.columns:
            labels = data['human_label']
        else:
            labels = [-1] * len(data)

        is_labelled = labels != -1
        y_labelled = labels[is_labelled]  
        X_labelled = features[is_labelled]

        # Allow the possibility of starting cold (no labels yet)
        if len(y_labelled) == 0:
            scores = [1] * len(data)
            acq = [1] * len(data)
        else:
            kernel = RBF() + WhiteKernel()
            self.estimator = GaussianProcessRegressor(kernel=kernel)
            self.estimator.fit(X_labelled, y_labelled)
            scores, std = self.estimator.predict(features, return_std=True)
            max_val = np.max(y_labelled)

            acq = self.EI(scores, std, max_val)

            if self.save_output:  # from PipelineStage superclass
                self.save_estimator()

        # duplicate score as trained_score so astronomaly understands it is 
        # also post-human
        return pd.DataFrame(
            index=features.index, 
            data=np.stack([scores, scores, acq], axis=1),  
            columns=['score', 'trained_score', 'acquisition']
        )

    def combine_data_frames(self, features, ml_df):
        """
        Convenience function to correctly combine dataframes.
        """
        return pd.concat((features, ml_df), axis=1, join='inner')

    def _execute_function(self, features_with_labels):
        """
        Does the work in actually running estimator.

        Parameters
        ----------
        data : pd.DataFrame or similar
            The anomalies dataframe that must include the column human_labels.

        Returns
        -------
        pd.DataFrame
            Contains the same original index of the features input and the 
            anomaly scores. 

        """

        num_labels = (features_with_labels['human_label'] != -1).sum()
        print(f'Running GP regression with {num_labels} training points')
        gp_output = self.update(features_with_labels)
        # print('GP min max',
        #       gp_output.trained_score.min(),
        #       gp_output.trained_score.max())
        return gp_output
