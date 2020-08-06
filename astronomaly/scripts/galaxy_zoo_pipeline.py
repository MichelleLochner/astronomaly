from astronomaly.data_management import image_reader
from astronomaly.preprocessing import image_preprocessing
from astronomaly.feature_extraction import power_spectrum, autoencoder
from astronomaly.feature_extraction import shape_features
from astronomaly.dimensionality_reduction import pca
from astronomaly.postprocessing import scaling
from astronomaly.anomaly_detection import isolation_forest, human_loop_learning
from astronomaly.anomaly_detection import lof
from astronomaly.visualisation import tsne
import os
import pandas as pd
import numpy as np

data_dir = '/home/michelle/BigData/Anomaly/'


window_size = 128
catalogue = None
band_prefixes = []
bands_rgb = {}
plot_cmap = 'hot'


image_dir = os.path.join(data_dir, 'GalaxyZoo', 
                         'galaxy-zoo-the-galaxy-challenge', 
                         'images_training_rev1', '')
output_dir = os.path.join(
    data_dir, 'astronomaly_output', 'images', 'galaxy_zoo', '')
image_transform_function = [
    image_preprocessing.image_transform_sigma_clipping,
    image_preprocessing.image_transform_scale]

display_transform_function = [
    image_preprocessing.image_transform_scale]

df = pd.read_csv(os.path.join(data_dir, 'GalaxyZoo', 
                              'galaxy-zoo-the-galaxy-challenge', 
                              'training_solutions_rev1.csv'),
                 index_col=0)
df.index = df.index.astype(str)
additional_metadata = df[['Class6.1']]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def run_pipeline():
    """
    An example of the full astronomaly pipeline run on image data

    Parameters
    ----------
    image_dir : str
        Directory where images are located (can be a single fits file or 
        several)
    features : str, optional
        Which set of features to extract on the cutouts
    dim_reduct : str, optional
        Which dimensionality reduction algorithm to use (if any)
    anomaly_algo : str, optional
        Which anomaly detection algorithm to use

    Returns
    -------
    pipeline_dict : dictionary
        Dictionary containing all relevant data including cutouts, features 
        and anomaly scores

    """

    # fls = os.listdir(image_dir)[:100]
    image_dataset = image_reader.ImageThumbnailsDataset(
        directory=image_dir, output_dir=output_dir, 
        transform_function=image_transform_function,
        display_transform_function=display_transform_function,
        # list_of_files=fls,
        additional_metadata=additional_metadata
    ) # noqa

    pipeline_ellipse = shape_features.EllipseFitFeatures(
        percentiles=[90, 80, 70, 60, 50, 0],
        output_dir=output_dir, channel=0, force_rerun=False, 
        central_contour=False)
    features_ellipse = pipeline_ellipse.run_on_dataset(image_dataset)

    # pipeline_psd = power_spectrum.PSD_Features(
    #     force_rerun=False, output_dir=output_dir)
    # features_psd = pipeline_psd.run_on_dataset(image_dataset)

    # pipeline_hu = shape_features.HuMomentsFeatures(
    #     sigma_levels=[2, 3, 4],
    #     output_dir=output_dir, channel=0, force_rerun=True)
    # features_hu = pipeline_hu.run_on_dataset(image_dataset)

    # features_original = features_ellipse.join(features_hu)
    # features_original = features_hu
    # features_ellipse = pd.read_parquet('/home/michelle/BigData/Anomaly/astronomaly_output/images/galaxy_zoo/EllipseFitFeatures_output.parquet')
    # features_psd = pd.read_parquet('/home/michelle/BigData/Anomaly/astronomaly_output/images/galaxy_zoo/PSD_Features_output_back_06_16.parquet')

    # features2 = pd.read_parquet('/home/michelle/BigData/Anomaly/astronomaly_output/images/galaxy_zoo/EllipseFitFeatures_output_back_06_16.parquet')
    # features = pd.read_parquet('/home/michelle/BigData/Anomaly/astronomaly_output/images/galaxy_zoo/EllipseFitFeatures_output_back_07_02.parquet')
    # features = features[features.columns[5:]]
    # new_features = features2[features2.columns[:5]].join(features)
    # features_ellipse = new_features

    features_original = features_ellipse
    # for col in features_original.columns:
    #     if "_0" in col or "_50" in col or "_60" in col:
    #         features_original = features_original.drop(col, axis=1)
    # features_original = features_psd
    features = features_original.copy()

    # pipeline_pca = pca.PCA_Decomposer(force_rerun=False, 
    #                                             output_dir=output_dir,
    #                                             threshold=0.95)
    # features = pipeline_pca.run(features_original)

    pipeline_scaler = scaling.FeatureScaler(force_rerun=False,
                                            output_dir=output_dir)
    features = pipeline_scaler.run(features)

    pipeline_iforest = isolation_forest.IforestAlgorithm(
        force_rerun=False, output_dir=output_dir)
    anomalies = pipeline_iforest.run(features)

    # pipeline_lof = lof.LOF_Algorithm(
    #     force_rerun=False, output_dir=output_dir)
    # anomalies = pipeline_lof.run(features)

    pipeline_score_converter = human_loop_learning.ScoreConverter(
        force_rerun=False, output_dir=output_dir)
    anomalies = pipeline_score_converter.run(anomalies)
    anomalies = anomalies.sort_values('score', ascending=False)

    N_labels = 200
    anomalies['human_label'] = [-1] * len(anomalies)
    inds = anomalies.index[:N_labels]
    human_probs = additional_metadata.loc[inds, 'Class6.1']
    human_scores = np.round(human_probs * 5).astype('int')
    anomalies.loc[inds, 'human_label'] = human_scores.astype(int)

    try:
        if 'human_label' not in anomalies.columns:
            df = pd.read_csv(
                os.path.join(output_dir, 'ml_scores.csv'), 
                index_col=0,
                dtype={'human_label': 'int'})
            df.index = df.index.astype('str')

            if len(anomalies) == len(df):
                anomalies = pd.concat(
                    (anomalies, df['human_label']), axis=1, join='inner')
    except FileNotFoundError:
        pass

    pipeline_active_learning = human_loop_learning.NeighbourScore(
        alpha=1, output_dir=output_dir)

    pipeline_tsne = tsne.TSNE_Plot(
        force_rerun=False,
        output_dir=output_dir,
        perplexity=50)
    t_plot = pipeline_tsne.run(features.loc[anomalies.index])
    # t_plot = np.log(features_scaled + np.abs(features_scaled.min())+0.1)

    return {'dataset': image_dataset, 
            'features': features, 
            'anomaly_scores': anomalies,
            'visualisation': t_plot, 
            'active_learning': pipeline_active_learning}


# run_pipeline(image_dir='/home/michelle/BigData/Anomaly/Meerkat_deep2/')
