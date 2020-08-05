from astronomaly.base.base_pipeline import PipelineStage
import numpy as np
import pandas as pd
from os import path


class PCA_Decomposer(PipelineStage):
    def __init__(self, n_components=0, threshold=0, **kwargs):
        """
        Dimensionality reduction with principle component analysis (PCA). 
        Wraps the scikit-learn function.

        Parameters
        ----------
        n_components : int
            Requested number of principle components to use. If 0 (default), 
            returns the maximum number of components.
        threshold : float
            An alternative to n_components. Will use sufficient components to 
            ensure threshold explained variance is achieved. Scikit-learn uses 
            the kwarg n_components to specify either an int or float but we are
            explicit here.
        """
        super().__init__(n_components=n_components, threshold=threshold, 
                         **kwargs)

        self.n_components = n_components

        if self.n_components == 0:
            self.n_components = None
        if 0 < threshold < 1:
            self.n_components = threshold

        self.pca_obj = None

    def save_pca(self, features):
        """
        Stores the mean and components of the PCA to disk. Makes use of the 
        original features information to label the columns.

        Parameters
        ----------
        features : pd.DataFrame or similar
            The original feature set the PCA was run on.
        """
        if self.pca_obj is not None:
            mn = self.pca_obj.mean_
            comps = self.pca_obj.components_
            dat = np.vstack((mn, comps))
            index = ['mean']
            for i in range(len(comps)):
                index += ['component%d' % i]
            df = pd.DataFrame(data=dat, columns=features.columns, index=index)

            self.save(df, path.join(self.output_dir, 'pca_components'))

    def _execute_function(self, features):
        """
        Actually does the PCA reduction and returns a dataframe.
        """
        from sklearn.decomposition import PCA

        self.pca_obj = PCA(self.n_components)
        self.pca_obj.fit(features)

        print('Total explained variance:', 
              np.sum(self.pca_obj.explained_variance_ratio_))

        output = self.pca_obj.transform(features)

        if self.save_output:
            self.save_pca(features)

        return pd.DataFrame(data=output, index=features.index)
