# Example of astronomaly applied to a fits image
from astronomaly.data_management import image_reader
from astronomaly.preprocessing import image_preprocessing
from astronomaly.feature_extraction import power_spectrum
from astronomaly.feature_extraction import shape_features
from astronomaly.dimensionality_reduction import pca
from astronomaly.postprocessing import scaling
from astronomaly.anomaly_detection import isolation_forest, human_loop_learning
from astronomaly.visualisation import tsne
import os
import pandas as pd


# Root directory for data
data_dir = os.path.join(os.path.sep, 'home', 'michelle', 'BigData', 
                        'Anomaly', '')

image_dir = os.path.join(data_dir, 'GOODS_S/', 'combined/')
output_dir = os.path.join(
    data_dir, 'astronomaly_output', 'images', 'goods', '')
window_size = 128
catalogue = pd.read_csv(
    os.path.join(image_dir, 'h_goods_sz_r2.0z_cat.csv'))
band_prefixes = ['iz-', 'v-', 'b-']
bands_rgb = {'r': 'iz-', 'g': 'v-', 'b': 'b-'}
plot_cmap = 'bone'
feature_method = 'ellipse'
dim_reduction = ''

if not os.path.exists(image_dir):
    os.makedirs(image_dir)

if len(os.listdir(image_dir)) == 0:
    # No data to run on!
    print('No data found to run on, downloading some GOODS-S data...')
    os.system(
        "wget " + # noqa
        "https://archive.stsci.edu/pub/hlsp/goods/v2/" +
        "h_sb_sect23_v2.0_drz_img.fits " + 
        "-P " + image_dir 
        )
    print('GOODS-S data downloaded.')


image_transform_function = [image_preprocessing.image_transform_sigma_clipping,
                            image_preprocessing.image_transform_scale]

display_transform_function = [image_preprocessing.image_transform_scale]


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
    image_dataset = image_reader.ImageDataset(
            directory=image_dir,
            window_size=window_size, output_dir=output_dir, plot_square=False,
            transform_function=image_transform_function,
            display_transform_function=display_transform_function,
            plot_cmap=plot_cmap,
            catalogue=catalogue,
            band_prefixes=band_prefixes,
            bands_rgb=bands_rgb
            ) # noqa

    if feature_method == 'psd':
        pipeline_psd = power_spectrum.PSD_Features(
            force_rerun=True, output_dir=output_dir)
        features_original = pipeline_psd.run_on_dataset(image_dataset)

    elif feature_method == 'ellipse':
        pipeline_ellipse = shape_features.EllipseFitFeatures(
            percentiles=[90, 80, 70, 60, 50, 0],
            output_dir=output_dir, channel=0, force_rerun=False
        )
        features_original = pipeline_ellipse.run_on_dataset(image_dataset)

    features = features_original.copy()

    if dim_reduction == 'pca':
        pipeline_pca = pca.PCA_Decomposer(force_rerun=False, 
                                          output_dir=output_dir,
                                          threshold=0.95)
        features = pipeline_pca.run(features_original)

    pipeline_scaler = scaling.FeatureScaler(force_rerun=False,
                                            output_dir=output_dir)
    features = pipeline_scaler.run(features)

    pipeline_iforest = isolation_forest.IforestAlgorithm(
        force_rerun=False, output_dir=output_dir)
    anomalies = pipeline_iforest.run(features)

    pipeline_score_converter = human_loop_learning.ScoreConverter(
        force_rerun=False, output_dir=output_dir)
    anomalies = pipeline_score_converter.run(anomalies)
    anomalies = anomalies.sort_values('score', ascending=False)

    try:
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

    return {'dataset': image_dataset, 
            'features': features, 
            'anomaly_scores': anomalies,
            'visualisation': t_plot, 
            'active_learning': pipeline_active_learning}
