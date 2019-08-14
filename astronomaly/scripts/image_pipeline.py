from astronomaly.data_management.image_reader import read_images, read_cutouts
from astronomaly.preprocessing import image_preprocessing
from astronomaly.feature_extraction import power_spectrum, wavelet_features
from astronomaly.dimensionality_reduction import decomposition
from astronomaly.anomaly_detection import isolation_forest, human_loop_learning, read_from_file
from astronomaly.clustering import tsne
from astronomaly.postprocessing import scaling


def run_pipeline(image_dir='', cutouts_file='', features = 'psd', dim_reduct = 'pca', 
                scaled = 'scaled', anomaly_algo = 'iforest', clustering='tsne',
                nproc=1, anomaly_file=''):
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

    if len(image_dir) != 0:
        pipeline_dict = read_images(image_dir)
        pipeline_dict = image_preprocessing.generate_cutouts(pipeline_dict, window_size=128,
                                                        transform_function=image_preprocessing.image_transform_log)
    elif len(cutouts_file) != 0:
        pipeline_dict = read_cutouts(cutouts_file)
    else:
        print('Either images or cutouts must be provided')
        #Raise some error
    if features == 'psd2d':
        pipeline_dict = power_spectrum.extract_features_psd2d(pipeline_dict, nbins='auto', 
                        nproc=nproc)
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
    
    if scaled == 'scaled':
        pipeline_dict = scaling.scale_features(pipeline_dict, input_key)
        input_key = input_key + '_scaled'

    if anomaly_algo == 'iforest':
        pipeline_dict = isolation_forest.run_isolation_forest(pipeline_dict, input_key,
                                                              output_column_name='iforest_score')
    pipeline_dict = human_loop_learning.convert_anomaly_score(pipeline_dict, 'iforest_score',
                                                              output_column='anomaly_score')

    if anomaly_file != '':
        pipeline_dict = read_from_file.read_anomaly_score(pipeline_dict, anomaly_file)

    if clustering == 'tsne':
        pipeline_dict = tsne.make_tsne(pipeline_dict, input_key, sort_by_column='anomaly_score',
        perplexity=50)


    

    return pipeline_dict


