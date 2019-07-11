import numpy as np
from sklearn.ensemble.iforest import IsolationForest
import pandas as pd


def run_isolation_forest(pipeline_dict, input_key, output_column_name='', contamination='auto'):
    """
    Runs sklearn's isolation forest anomaly detection algorithm and stores the final scores in the 'ml_scores' key
    in the pipeline_dict as a pandas dataframe.

    Parameters
    ----------
    pipeline_dict : dict
        Dictionary containing all relevant data including cutouts, features and anomaly scores
    input_key : str
        The input key of pipeline_dict to run the function on.
    output_column_name : str, optional
        The output column name of the 'ml_scores' dataframe in pipeline_dict.
        If not provided defaults to 'iforest_score_'+input_key
    contamination : string or float, optional
        Hyperparameter to pass to IsolationForest. 'auto' is recommended

    Returns
    -------
    pipeline_dict : dict
        Dictionary containing all relevant data including cutouts, features and anomaly scores

    """
    print('Running isolation forest...')
    if output_column_name == '':
        output_column_name = 'iforest_score_'+input_key
    feats = pipeline_dict[input_key]
    iforest = IsolationForest(contamination=contamination, behaviour='new')
    iforest.fit(feats)

    scores = iforest.decision_function(feats)

    df = pipeline_dict['metadata']
    if 'ml_scores' in pipeline_dict:
        pipeline_dict['ml_scores'][output_column_name] = scores
    else:
        pipeline_dict['ml_scores'] = pd.DataFrame({'id':df.id, output_column_name: scores})

    pipeline_dict['iforest_object_'+input_key] = iforest

    print('Done!')

    return pipeline_dict


