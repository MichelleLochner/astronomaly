from astronomaly.data_management.image_reader import read_images
from astronomaly.preprocessing import image_preprocessing
from astronomaly.feature_extraction import power_spectrum, wavelet_features
from astronomaly.dimensionality_reduction import decomposition
from astronomaly.anomaly_detection import isolation_forest, human_loop_learning


def run_pipeline(image_dir, features = 'psd', dim_reduct = 'pca', anomaly_algo = 'iforest'):
    """
    An example of the full astronomaly pipeline run on image data

    Parameters
    ----------
    image_dir : str
        Directory where images are located (can be a single fits file or several)
    features : str, optional
        Which set of features to extract on the cutouts
    dim_reduct : str, optional
        Which dimensionality reduction algorithm to use (if any)
    anomaly_algo : str, optional
        Which anomaly detection algorithm to use

    Returns
    -------
    pipeline_dict : dictionary
        Dictionary containing all relevant data including cutouts, features and anomaly scores

    """

    pipeline_dict = read_images(image_dir)
    pipeline_dict = image_preprocessing.generate_cutouts(pipeline_dict, window_size=128,
                                                        transform_function=image_preprocessing.image_transform_log)

    if features == 'psd':
        pipeline_dict = power_spectrum.extract_features_psd2d(pipeline_dict, nbins='auto')
        input_key = 'features_psd2d'
    elif features == 'wavelets':
        pipeline_dict = wavelet_features.extract_features_wavelets(pipeline_dict)
        input_key = 'features_wavelets'

    if dim_reduct == 'pca':
        pipeline_dict = decomposition.pca_decomp(pipeline_dict, input_key, n_components=10)
        input_key = input_key+'_pca'
    elif dim_reduct == 'trunc_svd':
        pipeline_dict = decomposition.truncated_svd_decomp(pipeline_dict, input_key, 20)
        input_key = input_key + '_trunc_svd'

    if anomaly_algo == 'iforest':
        pipeline_dict = isolation_forest.run_isolation_forest(pipeline_dict, input_key,
                                                              output_column_name='iforest_score')


    pipeline_dict = human_loop_learning.convert_anomaly_score(pipeline_dict, 'iforest_score',
                                                              output_column='anomaly_score')

    return pipeline_dict


