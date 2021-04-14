import logging

from sklearn.manifold import TSNE
import numpy as np
import pandas as pd

from astronomaly.base.base_pipeline import PipelineStage


class UMap(PipelineStage):
    def __init__(self, max_samples=1000, shuffle=False, 
                 **kwargs):
        """
        Rescales features using a standard sklearn scalar that subtracts the 
        mean and divides by the standard deviation for each feature. Highly 
        recommended for most machine learning algorithms and for any data 
        visualisation such as t-SNE.

        Parameters
        ----------
        perplexity : float, optional
            The perplexity is related to the number of nearest neighbors that  
            is used in other manifold learning algorithms (see t-SNE
            documentation), by default 30
        max_samples : int, optional
            Limits the computation to this many samples (by default 2000). Will
            be the first 2000 samples if shuffle=False. This is very useful as
            t-SNE scales particularly badly with sample size.
        shuffle : bool, optional
            Randomises the sample before selecting max_samples, by default 
            False
        """
        super().__init__(**kwargs)
        self.shuffle = shuffle

    def _execute_function(self, features):
        """
        Does the work in actually running the pipeline stage.

        Parameters
        ----------
        features : pd.DataFrame or similar
            The input features to run on. Assumes the index is the id 
            of each object and all columns are to be used as features.

        Returns
        -------
        pd.DataFrame
            Returns a dataframe with the same index as the input features and
            two columns, one for each dimension of the t-SNE plot.

        """
        
        if len(features.columns.values) == 2:
            logging.warning('Already dim 2 - skipping umap')
            return features.copy()

        # if len(features) > self.max_samples:
        #     if not self.shuffle:
        #         inds = features.index[:self.max_samples]
        #     else:
        #         inds = np.random.choice(features.index, self.max_samples, 
        #                                 replace=False)
        #     features = features.loc[inds]

        # ts = TSNE(perplexity=self.perplexity, learning_rate=10, n_iter=5000)
        # ts.fit(features)

        # fitted_tsne = ts.embedding_

        # return pd.DataFrame(data=fitted_tsne, index=features.index)
