

def convert_anomaly_score(pipeline_dict, input_column, output_column='', lower_is_weirder=True,
                          new_min=0, new_max=5, convert_integer=False):
    """
    Convenience function to convert anomaly scores onto a standardised scale, for use with the human-in-the-loop
    labelling frontend.

    Parameters
    ----------
    pipeline_dict : dict
        Dictionary containing all relevant data including cutouts, features and anomaly scores
    input_column : str
        The input column of the 'ml_scores' dataframe inside pipeline_dict to run the function on.
    output_column : str, optional
        The output input column of the 'ml_scores' dataframe inside pipeline_dict.
        If not provided defaults to input_column + '_norm
    lower_is_weirder : bool, optional
        If true, it means the anomaly scores in input_column correspond to a lower is more anomalous system, such as
        output by isolation forest.
    new_min : int or float, optional
        The new minimum score (now corresponding to the most boring objects)
    new_max : int or float, optional
        The new maximum score (now corresponding to the most interesting objects)
    convert_integer : bool, optional
        If true will force the resulting scores to be integer.

    Returns
    -------
    pipeline_dict : dict
        Dictionary containing all relevant data including cutouts, features and anomaly scores

    """
    if output_column == '':
        output_column = input_column+'_norm'
    df = pipeline_dict['ml_scores']

    scores = df[input_column]

    if lower_is_weirder:
        scores = -scores

    scores = (new_max - new_min)*(scores-scores.min())/(scores.max()-scores.min())+new_min

    if convert_integer:
        scores = round(scores)

    df[output_column] = scores.astype(int)

    return pipeline_dict

