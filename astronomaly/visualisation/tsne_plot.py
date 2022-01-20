from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from astronomaly.base.base_pipeline import PipelineStage


class TSNE_Plot(PipelineStage):
    def __init__(self, perplexity=30, max_samples=2000, shuffle=False, 
                 **kwargs):
        """
        Computes a t-SNE 2d visualisation of the data

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
        super().__init__(perplexity=perplexity, max_samples=max_samples, 
                         shuffle=shuffle, **kwargs)
        self.perplexity = perplexity
        self.max_samples = max_samples
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

        if len(features) > self.max_samples:
            if not self.shuffle:
                inds = features.index[:self.max_samples]
            else:
                inds = np.random.choice(features.index, self.max_samples, 
                                        replace=False)
            features = features.loc[inds]

        ts = TSNE(perplexity=self.perplexity, learning_rate=10, n_iter=5000)
        ts.fit(features)

        fitted_tsne = ts.embedding_

        return pd.DataFrame(data=fitted_tsne, index=features.index)
