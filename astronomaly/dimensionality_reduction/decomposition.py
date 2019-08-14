from sklearn.decomposition import PCA
from sklearn.decomposition.truncated_svd import TruncatedSVD
import numpy as np
import xarray


def pca_decomp(pipeline_dict, input_key, output_key='', n_components=0, threshold=0):
    """
    Perform standard PCA decomposition. Also stores the created PCA object in the pipeline_dict to allow quick
    transform of future data.

    Parameters
    ----------
    pipeline_dict : dict
        Dictionary containing all relevant data including cutouts, features and anomaly scores
    input_key : str
        The input key of pipeline_dict to run the function on. This specifies which set of features to use.
    output_key : str, optional
        The output key of pipeline_dict. If not set, will default to input_key+'_pca'
    n_components : int, optional
        How many components to return. If 0 will return all components (no dimensionality reduction)
    threshold : float, optional
        Use the energy to decide how many components to use. Will return enough components such that the total explained
        variance = threshold. (Overrides n_components)

    Returns
    -------
    pipeline_dict : dict
        Dictionary containing all relevant data including cutouts, features and anomaly scores

    """
    if len(output_key) == 0:
        output_key = input_key+'_pca'

    print('Running PCA...')
    feats = pipeline_dict[input_key]
    if n_components == 0:
        n_components = None
    if 0<threshold<1:
        n_components = threshold
    pca_obj = PCA(n_components)
    pca_obj.fit(feats)

    print('Total explained variance:', np.sum(pca_obj.explained_variance_ratio_))

    output = pca_obj.transform(feats)

    pipeline_dict[output_key] = xarray.DataArray(
        output, coords={'id':pipeline_dict['metadata'].id}, dims=['id', 'features'], name=output_key)

    pipeline_dict[input_key+'_pca_object'] = pca_obj

    return pipeline_dict


def truncated_svd_decomp(pipeline_dict, input_key, n_components, output_key=''):
    """
    Perform a truncated SVD decomposition. This is very useful for extremely high dimensional data (>10000 features)
    although it's not guaranteed to return the same coefficients each run. Also stores the created object in the
    pipeline_dict to allow quick transform of future data.

    Parameters
    ----------
    pipeline_dict : dict
        Dictionary containing all relevant data including cutouts, features and anomaly scores
    input_key : str
        The input key of pipeline_dict to run the function on. This specifies which set of features to use.
    n_components : int
        How many components to return.
    output_key : str, optional
        The output key of pipeline_dict. If not set, will default to input_key+'_trunc_svd'

    Returns
    -------
    pipeline_dict : dict
        Dictionary containing all relevant data including cutouts, features and anomaly scores

    """
    if len(output_key) == 0:
        output_key = input_key + '_trunc_svd'

    print('Running truncated SVD...')
    feats = pipeline_dict[input_key]
    trunc_svd = TruncatedSVD(n_components)
    trunc_svd.fit(feats)

    print('Total explained variance:', np.sum(trunc_svd.explained_variance_ratio_))

    output = trunc_svd.transform(feats)
    pipeline_dict[output_key] = xarray.DataArray(
        output, coords={'id': pipeline_dict['metadata'].id}, dims=['id', 'features'], name=output_key)

    pipeline_dict[input_key+'_trunc_svd_object'] = trunc_svd

    return pipeline_dict
