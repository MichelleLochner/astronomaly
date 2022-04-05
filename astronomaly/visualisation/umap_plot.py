import umap
import numpy as np
import pandas as pd

from astronomaly.base.base_pipeline import PipelineStage
from astronomaly.base import logging_tools


class UMAP_Plot(PipelineStage):
    # https://umap-learn.readthedocs.io/en/latest/api.html
    def __init__(self, min_dist=0.1, n_neighbors=15, max_samples=2000,
                 shuffle=False, **kwargs):
        """
        Computes a UMAP visualisation of the data

        Parameters
        ----------
        min_dist: float (optional, default 0.1)
            (Taken from UMAP documentation)
            The effective minimum distance between embedded points. Smaller
            values will result in a more clustered/clumped embedding where
            nearby points on the manifold are drawn closer together, while
            larger values will result on a more even dispersal of points. The
            value should be set relative to the spread value, which determines
            the scale at which embedded points will be spread out.
        n_neighbors: float (optional, default 15)
            (Taken from UMAP documentation)
            The size of local neighborhood (in terms of number of neighboring
            sample points) used for manifold approximation. Larger values
            result in more global views of the manifold, while smaller values
            result in more local data being preserved. In general values
            should be in the range 2 to 100.
        max_samples : int, optional
            Limits the computation to this many samples (by default 2000). Will
            be the first 2000 samples if shuffle=False. This is very useful as
            t-SNE scales particularly badly with sample size.
        shuffle : bool, optional
            Randomises the sample before selecting max_samples, by default 
            False
        """
        super().__init__(min_dist=min_dist, n_neighbors=n_neighbors,
                         max_samples=max_samples, shuffle=shuffle, **kwargs)
        self.max_samples = max_samples
        self.shuffle = shuffle
        self.min_dist = min_dist
        self.n_neighbors = n_neighbors

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
            two columns, one for each dimension of the UMAP plot.
        """

        if len(features.columns.values) == 2:
            logging_tools.log('Already dim 2 - skipping umap', level='WARNING')
            return features.copy()

        # copied from tsne
        if len(features) > self.max_samples:
            if not self.shuffle:
                inds = features.index[:self.max_samples]
            else:
                inds = np.random.choice(features.index, self.max_samples, 
                                        replace=False)
            features = features.loc[inds]

        reducer = umap.UMAP(n_components=2, min_dist=self.min_dist,
                            n_neighbors=self.n_neighbors)
        logging_tools.log('Beginning umap transform')
        reduced_embed = reducer.fit_transform(features)
        logging_tools.log('umap transform complete')

        return pd.DataFrame(data=reduced_embed, index=features.index)
