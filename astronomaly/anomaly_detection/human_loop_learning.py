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
            Default is 'score'. If 'all' is used, will convert all columns 
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

        scores = rescale_array(scores, self.new_min, self.new_max)
        return scores

# refactored for use elsewhere
def rescale_array(array, new_min, new_max, convert_integer=False):

        rescaled = (new_max - new_min) * (array - array.min()) / \
            (array.max() - array.min()) + new_min

        if convert_integer:
            rescaled = round(rescaled)

        return rescaled

class NeighbourScore(PipelineStage):
    def __init__(self, min_score=0.1, max_score=5, alpha=1, **kwargs):
        """
        Computes a new anomaly score based on what the user has labelled,
        allowing anomalous but boring objects to be rejected. This function
        takes training data (in the form of human given labels) and then
        performs regression to be able to predict user scores as a function of
        feature space. In regions of feature space where the algorithm is
        uncertain (i.e. there was little training data), it simply returns
        close to the original anomaly score. In regions where there was more
        training data, the anomaly score is modulated by the predicted user
        score resulting in the user seeing less "boring" objects.

        Parameters
        ----------
        min_score : float
            The lowest user score possible (must be greater than zero)
        max_score : float
            The highest user score possible
        alpha : float
            A scaling factor of how much to "trust" the predicted user scores.
            Should be close to one but is a tuning parameter.
        """
        super().__init__(min_score=min_score, max_score=max_score, alpha=alpha, 
                         **kwargs)
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
        nearest_neighbour_distance : array
            The distance of each instance to its nearest labelled neighbour.
        user_score : array
            The predicted user score for each instance
        anomaly_score : array
            The actual anomaly score from a machine learning algorithm

        Returns
        -------
        array
            The final anomaly score for each instance, penalised by the
            predicted user score as required.
        """
        return weight_by_distance(nearest_neighbour_distance, user_score, anomaly_score, min_score=self.min_score, max_score=self.max_score, alpha=self.alpha)



    def compute_nearest_neighbour(self, features_with_labels):
        """
        Calculates the distance of each instance to its nearest labelled
        neighbour. 

        Parameters
        ----------
        features_with_labels : pd.DataFrame
            A dataframe where the first columns are the features  and the last
            two columns are 'human_label' and 'score' (the anomaly score from
            the ML algorithm).

        Returns
        -------
        array
            Distance of each instance to its nearest labelled neighbour.
        """
        features = features_with_labels.drop(columns=['human_label', 'score', 'trained_score', 'acquisition'], errors='ignore')  # may not all be present

        label_mask = ~pd.isna(features_with_labels['human_label'])
        labelled = features.loc[features_with_labels.index[label_mask]].values
        features = features.values

        return get_distances(labelled, features)


    def train_regression(self, features_with_labels):
        """
        Uses machine learning to predict the user score for all the data. The
        labels are provided in the column 'human_label' which must be -1 if no
        label exists.

        Parameters
        ----------
        features_with_labels : pd.DataFrame
            A dataframe where the first columns are the features  and the last
            two columns are 'human_label' and 'score' (the anomaly score from
            the ML algorithm).

        Returns
        -------
        array
            The predicted user score for each instance.
        """
        label_mask = ~pd.isna(features_with_labels['human_label'])
        inds = features_with_labels.index[label_mask]
        features = features_with_labels.drop(columns=['human_label', 'score', 'trained_score', 'acquisition'], errors='ignore')
        reg = RandomForestRegressor(n_estimators=100)
        reg.fit(features.loc[inds], 
                features_with_labels.loc[inds, 'human_label'])

        fitted_scores = reg.predict(features)
        return fitted_scores


    def _execute_function(self, features_with_labels):
        """
        Does the work in actually running the NeighbourScore.

        Parameters
        ----------
        features_with_labels : pd.DataFrame
            A dataframe where the first columns are the features  and the last
            two columns are 'human_label' and 'score' (the anomaly score from
            the ML algorithm).

        Returns
        -------
        pd.DataFrame
            Contains the final scores using the same index as the input.

        """
        distances = self.compute_nearest_neighbour(features_with_labels)
        regressed_score = self.train_regression(features_with_labels)
        # combine weighted towards regressed score where distances are small and vica versa
        trained_score = self.anom_func(distances, 
                                       regressed_score, 
                                       features_with_labels.score.values)
        # df with features.index as index and trained_score column
        # I will replace with gp that gives trained_score
        return pd.DataFrame(data=trained_score, 
                            index=features_with_labels.index, 
                            columns=['trained_score'])

# refactored to use in ml_comparison.py
def weight_by_distance(nearest_neighbor_distance, user_score, anomaly_score, min_score, max_score, alpha):
    # u^tilde
    f_u = min_score + 0.85 * (user_score / max_score)  # e2 = 0.85
    # d0 = euclidean distance in feature space to a labelled neighbour
    # d = mean d0 for whole dataset
    d0 = nearest_neighbor_distance / np.mean(nearest_neighbor_distance)  # mostly 0-1 with a few outliers dragging the mean up
    # d0 = nearest_neighbor_distance / np.median(nearest_neighbor_distance)  # median might work better with outliers?
    
    # clip before exponentiating might work better, some are huge
    # d0 = np.clip(d0, 0, 10)

    dist_penalty = np.exp(d0 * alpha)  # around 1.5 to 2
    
    # might work better to drop the exponential
    # dist_penalty = d0 * alpha

    # arctanh f_u around 1
    # eqn 1. Close to 2, becomes 1, and most are ~2
    # will always be <= 1 i.e. user votes will only ever downweight isolationforest results
    user_weights = np.tanh(dist_penalty - 1 + np.arctanh(f_u))  
    return anomaly_score * user_weights  

def get_distances(labelled_points, query_points):
    # returns list of distances between query point[i] and closest labelled point, for all i
    # TODO modified by mike to exclude distance to self

    # for whatever reason, this breaks everything

    # tree = cKDTree(labelled_points)
    # distances = np.zeros(len(query_points))
    # for i in range(len(query_points)):
    #     distances_to_query_point, _ = tree.query(query_points[i], k=3)
    #     if distances_to_query_point[0] == 0:  # distance to itself
    #         dist = distances_to_query_point[1]
    #     else:
    #         dist = distances_to_query_point[0]
    #     distances[i] = dist
    # # print(labelled_points)
    # return distances

    # stick to the original

    mytree = cKDTree(labelled_points)
    distances = np.zeros(len(query_points))
    for i in range(len(query_points)):
        dist = mytree.query(query_points[i])[0]
        distances[i] = dist
    # print(labelled_points)
    return distances
