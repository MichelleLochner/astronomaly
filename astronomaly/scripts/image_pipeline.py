from astronomaly.data_management import image_reader
from astronomaly.preprocessing import image_preprocessing
from astronomaly.feature_extraction import power_spectrum, autoencoder
from astronomaly.dimensionality_reduction import decomposition
from astronomaly.postprocessing import scaling
from astronomaly.anomaly_detection import isolation_forest, human_loop_learning
from astronomaly.clustering import tsne
import os
import pandas as pd

data_dir = '/home/michelle/BigData/Anomaly/'
# data_dir = './'

# which_data = 'meerkat'
# which_data = 'meerkat_deep2'
# which_data = 'goods'
# which_data = 'tgss'
which_data = 'decals'

window_size = 128
image_transform_function = image_preprocessing.image_transform_log
catalogue = None

if which_data == 'meerkat':
    image_dir = os.path.join(data_dir, 'Meerkat_data', 'Clusters')
    output_dir = os.path.join(
        data_dir, 'astronomaly_output', 'images', 'meerkat', '')
    plot_cmap = 'hot'
elif which_data == 'meerkat_deep2':
    image_dir = os.path.join(data_dir, 'Meerkat_data', 'meerkat_deep2')
    output_dir = os.path.join(
        data_dir, 'astronomaly_output', 'images', 'meerkat_deep2', '')
    plot_cmap = 'hot'

elif which_data == 'tgss':
    image_dir = os.path.join(data_dir, 'TGSS')
    output_dir = os.path.join(
        data_dir, 'astronomaly_output', 'images', 'tgss', '')
    plot_cmap = 'hot'
    window_size = 32

elif which_data == 'decals':
    image_dir = os.path.join(data_dir, 'decals')
    output_dir = os.path.join(
        data_dir, 'astronomaly_output', 'images', 'decals', '')
    plot_cmap = 'hot'
    window_size = 64
    catalogue = pd.read_csv(
        os.path.join(data_dir, 'decals', 'catalogues', 'tractor-0267m095.csv'))

else:
    image_dir = os.path.join(data_dir, 'GOODS_S/')
    output_dir = os.path.join(
        data_dir, 'astronomaly_output', 'images', 'goods', '')

    image_transform_function = image_preprocessing.image_transform_log

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    if len(os.listdir(image_dir)) == 0:
        # No data to run on!
        print('No data found to run on, downloading some GOODS-S data...')
        os.system(
            "wget " + # noqa
            "https://archive.stsci.edu/pub/hlsp/goods/v2/h_sb_sect23_v2.0_drz_img.fits " + # noqa
            '-P ' + image_dir 
            )
        print('GOODS-S data downloaded.')

    plot_cmap = 'bone'


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

feature_method = 'psd'
dim_reduction = 'pca'


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
        plot_cmap=plot_cmap,
        catalogue=catalogue
        ) # noqa

    if feature_method == 'psd':
        pipeline_psd = power_spectrum.PSD_Features(
            force_rerun=False, output_dir=output_dir)
        features_original = pipeline_psd.run_on_dataset(image_dataset)
        features = features_original.copy()

    elif feature_method == 'autoencoder':
        training_dataset = image_reader.ImageDataset(
            directory=image_dir, 
            transform_function=image_transform_function,
            window_size=window_size, window_shift=window_size // 2, 
            output_dir=output_dir)

        pipeline_autoenc = autoencoder.AutoencoderFeatures(
            output_dir=output_dir, training_dataset=training_dataset,
            retrain=True)
        features = pipeline_autoenc.run_on_dataset(image_dataset)

    if dim_reduction == 'pca':
        pipeline_pca = decomposition.PCA_Decomposer(force_rerun=False, 
                                                    output_dir=output_dir,
                                                    n_components=2)
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
    # t_plot = np.log(features_scaled + np.abs(features_scaled.min())+0.1)

    return {'dataset': image_dataset, 
            'features': features, 
            'anomaly_scores': anomalies,
            'cluster': t_plot, 
            'active_learning': pipeline_active_learning}


# run_pipeline(image_dir='/home/michelle/BigData/Anomaly/Meerkat_deep2/')
