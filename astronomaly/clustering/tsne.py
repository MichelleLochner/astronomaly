from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from astronomaly.base.base_pipeline import PipelineStage

class TSNE_Plot(PipelineStage):
    def __init__(self, perplexity=30, max_samples=2000, shuffle=False,**kwargs):
        """
        Rescales features using a standard sklearn scalar that subtracts the mean and divides by the standard deviation
        for each feature. Highly recommended for most machine learning algorithms and for any data visualisation such as
        t-SNE.
        """
        super().__init__(perplexity=perplexity, max_samples=max_samples, shuffle=shuffle, **kwargs)
        self.perplexity = perplexity
        self.max_samples = max_samples
        self.shuffle = shuffle

    def _execute_function(self, features):
        """
        Does the work in actually running the scaler.

        Parameters
        ----------
        features : pd.DataFrame or similar
            The input features to run iforest on. Assumes the index is the id of each object and all columns are to
            be used as features.

        Returns
        -------
        pd.DataFrame
            Contains the same original index and columns of the features input with the features scaled to zero mean
            and unit variance.

        """

        if len(features) > self.max_samples:
            if not self.shuffle:
                inds = features.index[:self.max_samples]
            else:
                inds = np.random.choice(features.index, self.max_samples, replace=False)
            features = features.loc[inds]


        ts = TSNE(perplexity=self.perplexity, learning_rate=10, n_iter=5000)
        ts.fit(features)

        fitted_tsne = ts.embedding_

        return pd.DataFrame(data=fitted_tsne, index=features.index)
        