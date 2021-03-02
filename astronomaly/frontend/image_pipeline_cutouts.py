from astronomaly.data_management import image_reader
from astronomaly.preprocessing import image_preprocessing
from astronomaly.feature_extraction import power_spectrum, autoencoder
from astronomaly.feature_extraction import shape_features
from astronomaly.dimensionality_reduction import pca
from astronomaly.postprocessing import scaling
from astronomaly.anomaly_detection import isolation_forest, human_loop_learning
from astronomaly.visualisation import tsne
from astronomaly.utils import utils
import os
import pandas as pd

#data_dir = '/home/verlon/Desktop/Files/Data/CUTOUTS/GT'
data_dir = '/home/verlon/Desktop/Files/Data/CUTOUTS/Test Set 15000'

which_data = 'decals'
list_of_files = []
window_size = 32

image_transform_function = [image_preprocessing.image_band_reorder,
                            image_preprocessing.image_transform_scale,
                            image_preprocessing.image_transform_greyscale,
                            #image_preprocessing.image_band_addition,
                            image_preprocessing.image_transform_sigma_clipping,
                            # image_preprocessing.image_transform_inverse_sinh,
                            image_preprocessing.image_transform_scale,
                            #image_preprocessing.image_transform_cv2_resize,
                            ]

display_transform_function = [#image_preprocessing.image_transform_inverse_sinh,
                              image_preprocessing.image_transform_scale
                              ]


image_dir = os.path.join(data_dir,'Cutouts')
output_dir = os.path.join(data_dir, 'Output', '')

#image_dir = os.path.join(data_dir,'0260m062', 'Input', 'Images')
#output_dir = os.path.join(data_dir,'0260m062', 'Output', '')

#catalogue = pd.read_csv(os.path.join(data_dir,'Ground Truth.csv'))
catalogue = pd.read_csv(os.path.join(data_dir,'Test Set 15000 (copy).csv'))
    #    '/home/verlon/Desktop/Astronomaly/Data/Coadd_0260/0260m062/Input/test_catalogue_0260m062_500.csv')
    #    os.path.join(data_dir, 'Images','z-legacysurvey-0260m062-image.fits.fz'),
    #    image_name = 'legacysurvey-0260m062-image.fits.fz')
band_prefixes = ['z-', 'r-', 'g-']
bands_rgb = {'r': 'z-', 'g': 'r-', 'b': 'g-'}
plot_cmap = 'hot'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

feature_method = 'ellipse'
dim_reduction = ''
extending_ellipse = False

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
    
    image_dataset = image_reader.ImageFitsDataset(
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
        #print(image_dataset.get_sample(image_dataset.index[0]))
        #print(image_dataset.images)


    pipeline_ellipse = shape_features.EllipseFitFeatures(
            percentiles=[90, 80, 70, 60, 50,0],
            output_dir=output_dir, channel=0,
            extending_ellipse = extending_ellipse,
            force_rerun=True
        )

    #features_original, contours, ellipses = pipeline_ellipse.run_on_dataset(image_dataset)
    features_original = pipeline_ellipse.run_on_dataset(image_dataset)

    features = features_original.copy()

    print(features)
    if extending_ellipse:
        features.drop(['Warning_Open_Ellipse'], 1, inplace=True)
    print(features)

    pipeline_scaler = scaling.FeatureScaler(force_rerun=True,
                                            output_dir=output_dir)
    features = pipeline_scaler.run(features)

    pipeline_iforest = isolation_forest.IforestAlgorithm(
        force_rerun=True, output_dir=output_dir)
    anomalies = pipeline_iforest.run(features)

    pipeline_score_converter = human_loop_learning.ScoreConverter(
        force_rerun=True, output_dir=output_dir)
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
        force_rerun=True,
        output_dir=output_dir,
        perplexity=50)
    t_plot = pipeline_tsne.run(features.loc[anomalies.index])
    # t_plot = np.log(features_scaled + np.abs(features_scaled.min())+0.1)

    flname = os.path.join(output_dir, 'anomaly_catalogue_all.xlsx')
    utils.create_catalogue_spreadsheet(image_dataset, anomalies[:200],
                                       filename=flname,
                                       ignore_nearby_sources=True,
                                       source_radius=0.016)

    image_dataset.fits_to_png(anomalies)

    return {'dataset': image_dataset, 
            'features': features, 
            'anomaly_scores': anomalies,
            'visualisation': t_plot, 
            'active_learning': pipeline_active_learning}


# run_pipeline(image_dir='/home/michelle/BigData/Anomaly/Meerkat_deep2/')
