from astronomaly.data_management import image_reader
from astronomaly.preprocessing import image_preprocessing
from astronomaly.feature_extraction import shape_features, power_spectrum
from astronomaly.dimensionality_reduction import pca
from astronomaly.postprocessing import scaling
from astronomaly.anomaly_detection import isolation_forest, human_loop_learning
from astronomaly.visualisation import tsne
from astronomaly.utils import utils
import os
import pandas as pd

data_dir = '/home/michelle/BigData/Anomaly/'

image_dir = os.path.join(data_dir, 'Meerkat_data', 'Clusters_legacy')
# list_of_files = ['J0232.2-4420.Fix.1pln.fits.gz']
fls = os.listdir(image_dir)
to_leave_out = ['Abell_168.1pln.fits.gz', 'Abell_2811B.1pln.fits.gz',
                'J2340.1-8510.Fix.1pln.fits.gz',
                'Abell_4038.Fix.1pln.fits.gz']
list_of_files = [f for f in fls if f not in to_leave_out]
output_dir = os.path.join(
    data_dir, 'astronomaly_output', 'images', 'meerkat_clusters', '')

# cat_file = os.path.join(
#     data_dir, 'Meerkat_data', 'Clusters_legacy', 'Catalogues',
#     'J0232.2-4420_processed1.pybdsm.srl.fits')
cat_file = os.path.join(
    data_dir, 'Meerkat_data', 'Clusters_legacy', 'Catalogues',
    'Abell_S295.plane0.pybdsm.srl.fits')

# image_file = os.path.join(image_dir, list_of_files[0])
# catalogue = utils.convert_pydsf_catalogue(cat_file, image_file)
catalogue = None

window_size = 64
image_transform_function = [
    image_preprocessing.image_transform_sigma_clipping,
    image_preprocessing.image_transform_scale]

display_transform_function = [
    image_preprocessing.image_transform_inverse_sinh,
    image_preprocessing.image_transform_scale]

band_prefixes = []
bands_rgb = {}
plot_cmap = 'hot'
feature_method = 'ellipse'
dim_reduction = ''

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def run_pipeline():
    """
    An example of the full astronomaly pipeline run on image data

    Returns
    -------
    pipeline_dict : dictionary
        Dictionary containing all relevant data including cutouts, features 
        and anomaly scores

    """

    image_dataset = image_reader.ImageDataset(
        directory=image_dir,
        list_of_files=list_of_files,
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

    flname = os.path.join(output_dir, 'anomaly_catalogue.xlsx')
    utils.create_catalogue_spreadsheet(image_dataset, anomalies[:100],
                                       filename=flname,
                                       ignore_nearby_sources=True,
                                       source_radius=0.016)

    return {'dataset': image_dataset, 
            'features': features, 
            'anomaly_scores': anomalies,
            'cluster': t_plot, 
            'active_learning': pipeline_active_learning}
