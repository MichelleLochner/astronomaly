from astronomaly.base.base_pipeline import PipelineStage
from astronomaly.base import logging_tools
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


class ScoreConverter(PipelineStage):
    def __init__(self, lower_is_weirder=True, new_min=0, new_max=5, 
                 convert_integer=False, column_name='score',
                 **kwargs):
        """
        Convenience function to convert anomaly scores onto a standardised 
        scale, for use with the human-in-the-loop labelling frontend.

        Parameters
        ----------
        lower_is_weirder : bool
            If true, it means the anomaly scores in input_column correspond to 
            a lower is more anomalous system, such as output by isolation 
            forest.
        new_min : int or float
            The new minimum score (now corresponding to the most boring 
            objects)
        new_max : int or float
            The new maximum score (now corresponding to the most interesting 
            objects)
        convert_integer : bool
            If true will force the resulting scores to be integer.
        column_name : str
            The name of the column to convert to the new scoring method. 
            Default is 'scores'. If 'all' is used, will convert all columns 
            the DataFrame.
        """
        super().__init__(lower_is_weirder=lower_is_weirder, new_min=new_min, 
                         new_max=new_max, **kwargs)
        self.lower_is_weirder = lower_is_weirder
        self.new_min = new_min
        self.new_max = new_max
        self.convert_integer = convert_integer
        self.column_name = column_name

    def _execute_function(self, df):
        """
        Does the work in actually running the scaler.

        Parameters
        ----------
        df : pd.DataFrame or similar
            The input anomaly scores to rescale.

        Returns
        -------
        pd.DataFrame

        Contains the same original index and columns of the features input 
        with the anomaly score scaled according to the input arguments in 
        __init__.

        """
        print('Running anomaly score rescaler...')

        if self.column_name == 'all':
            cols = df.columns
        else:
            cols = [self.column_name]
        try:
            scores = df[cols]
        except KeyError:
            msg = 'Requested column ' + self.column_name + ' not available in \
                    input dataframe. No rescaling has been performed'
            logging_tools.log(msg, 'WARNING')
            return df

        if self.lower_is_weirder:
            scores = -scores

        scores = (self.new_max - self.new_min) * (scores - scores.min()) / \
            (scores.max() - scores.min()) + self.new_min

        if self.convert_integer:
            scores = round(scores)

        return scores


class NeighbourScore(PipelineStage):
    def __init__(self, min_score=0.1, max_score=5, alpha=1, **kwargs):
        """
        Convenience function to convert anomaly scores onto a standardised 
        scale, for use with the human-in-the-loop labelling frontend.

        Parameters
        ----------
        lower_is_weirder : bool
            If true, it means the anomaly scores in input_column correspond to 
            a lower is more anomalous system, such as output by isolation 
            forest.
        new_min : int or float
            The new minimum score (now corresponding to the most boring 
            objects)
        new_max : int or float
            The new maximum score (now corresponding to the most interesting 
            objects)
        convert_integer : bool
            If true will force the resulting scores to be integer.
        column_name : str
            The name of the column to convert to the new scoring method. 
            Default is 'scores'. If 'all' is used, will convert all columns in 
            the DataFrame.
        """
        super().__init__(min_score=0.1, max_score=5, alpha=1, **kwargs)
        self.min_score = min_score
        self.max_score = max_score
        self.alpha = alpha

    def anom_func(self, nearest_neighbour_distance, user_score, anomaly_score):
        """
        Simple function that is dominated by the (predicted) user score in 
        regions where we are reasonably sure about our ability to predict that 
        score, and is dominated by the anomaly score from an algorithms in 
        regions we have little data.

        Parameters
        ----------
        """
        f_u = 0.1 + 0.85 * (user_score / 5)
        dist_penalty = np.exp(nearest_neighbour_distance / self.alpha)
        return anomaly_score * np.tanh(dist_penalty - 1 + np.arctanh(f_u))

    def compute_nearest_neighbour(self, features_with_labels):
        # Can definitely speed this up
        features = features_with_labels.drop(columns=['human_label', 'score'])
        # print(features)
        label_mask = features_with_labels['human_label'] != -1
        labelled = features.loc[features_with_labels.index[label_mask]].values
        features = features.values

        mytree = cKDTree(labelled)
        distances = np.zeros(len(features))
        for i in range(len(features)):
            dist, = mytree.query(features[i])
            distances[i] = dist
        # print(labelled)
        return distances

    def train_regression(self, features_with_labels):
        label_mask = features_with_labels['human_label'] != -1
        inds = features_with_labels.index[label_mask]
        features = features_with_labels.drop(columns=['human_label', 'score'])
        reg = RandomForestRegressor(n_estimators=100)
        reg.fit(features.loc[inds], 
                features_with_labels.loc[inds, 'human_label'])

        fitted_scores = reg.predict(features)
        return fitted_scores

    def combine_data_frames(self, features, ml_df):
        return pd.concat((features, ml_df), axis=1, join='inner')

    def _execute_function(self, features_with_labels):
        """
        Does the work in actually running the scaler.

        Parameters
        ----------
        df : pd.DataFrame or similar
            The input anomaly scores to rescale.

        Returns
        -------
        pd.DataFrame
            Contains the same original index and columns of the features input 
            with the anomaly score scaled according to the input arguments in 
            __init__.

        """

        distances = self.compute_nearest_neighbour(features_with_labels)
        retrained_score = self.train_regression(features_with_labels)
        final_score = self.anom_func(distances, 
                                     retrained_score, 
                                     features_with_labels.score.values)
        return pd.DataFrame(data=final_score, index=features_with_labels.index, 
                            columns=['final_score'])
