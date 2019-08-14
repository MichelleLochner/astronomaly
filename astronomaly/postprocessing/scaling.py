from sklearn.preprocessing import StandardScaler
import xarray

def scale_features(pipeline_dict, input_key, output_key=''):
    """
    Rescales features using a standard sklearn scalar that subtracts the mean and divides by the standard deviation
    for each feature. Highly recommended for most machine learning algorithms and for any data visualisation such as
    t-SNE.

    Parameters
    ----------
    pipeline_dict : dict
        Dictionary containing all relevant data including cutouts, features and anomaly scores
    input_key : str, optional
        The input key of pipeline_dict to run the function on.
    output_key : str, optional
        The output key of pipeline_dict

    Returns
    -------
    pipeline_dict : dict
        Dictionary containing all relevant data including cutouts, features and anomaly scores

    """
    scl = StandardScaler()
    output = scl.fit_transform(pipeline_dict[input_key])

    if output_key == '':
        output_key = input_key+'_scaled'
    pipeline_dict[output_key] = xarray.DataArray(
        output, coords={'id': pipeline_dict[input_key].id, 'features':pipeline_dict[input_key].features}, dims=['id', 'features'], name=output_key)

    return pipeline_dict
