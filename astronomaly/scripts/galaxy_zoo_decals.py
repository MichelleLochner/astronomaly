import os
import logging
import json
import glob

import pandas as pd
import numpy as np
import zipfile
from PIL import Image

logging.basicConfig(level=logging.INFO)

os.environ['FLASK_ENV'] = 'development'
# logging.basicConfig(level=logging.INFO)

from astronomaly.data_management import image_reader
from astronomaly.preprocessing import image_preprocessing
from astronomaly.feature_extraction import shape_features
from astronomaly.postprocessing import scaling
from astronomaly.anomaly_detection import isolation_forest, gaussian_process, human_loop_learning
from astronomaly.visualisation import tsne, umap


# Where output should be stored
output_dir = os.path.join('temp')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# image_transform_function = None
# TEMP for ellipse
image_transform_function = [
    lambda x: np.mean(x, axis=-1),  # mean across bands
    image_preprocessing.image_transform_sigma_clipping,
    image_preprocessing.image_transform_scale]


display_transform_function = None

max_galaxies = None

# # feature_loc = 'dr5_8_b0_pca2_and_safe_ids.parquet'  # 2D
# # feature_loc = 'dr5_8_b0_pca10_and_safe_ids.parquet  # 10D
# feature_loc = 'decals/cnn_features_concat.parquet'  # 10D
# feature_df = pd.read_parquet(feature_loc)
# feature_df['galaxy_id'] = feature_df['iauname']
# feature_df = feature_df[feature_df['galaxy_id'].str.startswith('J')]  # only the DR5 galaxies for now

# feature_cols = [col for col in feature_df.columns.values if col.startswith('feat')]
# features = feature_df[feature_cols + ['galaxy_id']]  # already pca'd
# features = features.set_index('galaxy_id')  # features must be string-indexed by objid, is not set to index internally, need to do manually
# features.index = features.index.rename('objid') # probably not needed

# full_catalog = pd.read_parquet('gz_decals_auto_posteriors.parquet')
# catalog = pd.merge(full_catalog, feature_df, on='galaxy_id', how='inner').reset_index(drop=True)

catalog = pd.read_csv('dr5_volunteer_catalog_internal.csv', usecols=['iauname', 'png_loc'])
# objid is set as index internally (but only after assign bug fixed)
catalog['objid'] = catalog['iauname'].astype(str) 
# catalog['objid'] = catalog['galaxy_id'].astype(str)
catalog['filename'] = catalog['png_loc']
# catalog['filename'] = catalog['iauname'].apply(lambda x: '/raid/scratch/walml/galaxy_zoo/decals/png/' + x[:4] + '/' + x + '.png')
catalog['filename'] = catalog['filename'].str.replace('/dr5', '/raid/scratch/walml/galaxy_zoo/decals/png')
# catalog['filename'] = catalog['filename'].str.replace('/media/walml/beta1/decals/png_native/dr5', '/raid/scratch/walml/galaxy_zoo/decals/png')
print(catalog['filename'][0])
assert os.path.isfile(catalog['filename'][0])


if max_galaxies is not None:
    catalog = catalog.sample(max_galaxies)
print(len(catalog))

# requires args like output_dir passed as globals, be careful
def run_pipeline():
    """
    Any script passed to the Astronomaly server must implement this function.
    run_pipeline must return a dictionary that contains the keys listed below.

    Parameters
    ----------

    Returns
    -------
    pipeline_dict : dictionary
        Dictionary containing all relevant data. Keys must include: 
        'dataset' - an astronomaly Dataset object
        'features' - pd.DataFrame containing the features
        'anomaly_scores' - pd.DataFrame with a column 'score' with the anomaly
        scores
        'visualisation' - pd.DataFrame with two columns for visualisation
        (e.g. TSNE or UMAP)
        'active_learning' - an object that inherits from BasePipeline and will
        run the human-in-the-loop learning when requested

    """

    # This creates the object that manages the data
    # order will be sorted alphabetically in directory, must match features
    image_dataset = image_reader.ImageThumbnailsDataset(
        catalogue=catalog[['filename', 'objid']],  # must have filename column. objid will be set as index.
        output_dir=output_dir, 
        transform_function=image_transform_function,
        display_transform_function=display_transform_function
    )

    """For debugging image loading"""
    # print(image_dataset.metadata)
    # print(image_dataset.metadata.index)
    # display_data = image_dataset.get_display_data(image_dataset.metadata.index[0])
    # im = Image.open(display_data).convert('RGB') 
    # im.show()

    """Temporarily added back to calculate ellipse features on decals"""
    pipeline_ellipse = shape_features.EllipseFitFeatures(
        percentiles=[90, 80, 70, 60, 50, 0],
        output_dir=output_dir, channel=0, force_rerun=False, 
        central_contour=False)
    features = pipeline_ellipse.run_on_dataset(image_dataset)
    pipeline_scaler = scaling.FeatureScaler(force_rerun=False,
                                            output_dir=output_dir)
    features = pipeline_scaler.run(features)
    features.to_parquet('decals_ellipse_features.parquet')  # index will be GalaxyID/objid
    exit()


    # The actual anomaly detection is called in the same way by creating an
    # Iforest pipeline object then running it

    # pipeline_iforest = isolation_forest.IforestAlgorithm(
    #     force_rerun=False, output_dir=output_dir)
    # anomalies = pipeline_iforest.run(features)
    # returns dataframe with "score" column and features.index as index

    
    try:
        # This is used by the frontend to store labels as they are applied so
        # that labels are not forgotten between sessions of using Astronomaly
        # includes 'human_label' and 'score' columns, for ml and humans respectively
        scores = pd.read_csv(
                    os.path.join(output_dir, 'ml_scores.csv'), 
                    index_col=0,
                    dtype={'human_label': float})
        scores.index = scores.index.astype('str')

        assert len(features) == len(scores)
        data = pd.concat(
            (features, scores), axis=1, join='outer')
        assert len(data) == len(features)
    except FileNotFoundError:
        # will save ml_scores.csv every time controller.get_label is called
        data = features.copy()
        data['human_label'] = np.nan
        data['score'] = np.nan
        data['trained_score'] = np.nan
        data['acquisition'] = np.nan

    # data = pd.concat([data, scores], axis=1)

    # We convert the scores onto a range of 0-5
    # pipeline_score_converter = human_loop_learning.ScoreConverter(
    #     force_rerun=False, output_dir=output_dir)
    # anomalies = pipeline_score_converter.run(anomalies)

    # anomalies df will now be modified to add human_label column if there are previous human_labels, saved (confusingly) in ml_scores.csv 
    # (because it's added as a column to anomalies df rather than its own csv...)


    # I thought this might subclass soemthing else with dif. requirements but no, it's just pipelinestage
    # This is the active learning object that will be run on demand by the
    # frontend 
    # pipeline_active_learning = human_loop_learning.NeighbourScore(
    #     alpha=1, output_dir=output_dir)

    # We use TSNE for visualisation which is run in the same way as other parts
    # of the pipeline.
    # pipeline_tsne = tsne.TSNE_Plot(
    #     force_rerun=False,
    #     output_dir=output_dir,
    #     perplexity=100)
    # visualisation = pipeline_tsne.run(features)

    # visualisation = None

    pipeline_umap = umap.UMap(max_samples=1000)
    visualisation = pipeline_umap.run(features)

    print(data[['human_label', 'score', 'acquisition']].head())

    gp_learning = gaussian_process.GaussianProcess(
        force_rerun=False, output_dir=output_dir
    )
    
    # anomaly_scores = pd.concat([human_labels, scores], axis=1)  # anomaly scores includes both ml and human scores
    # The run_pipeline function must return a dictionary with these keywords
    return {'dataset': image_dataset, 
            'features': features, 
            'anomaly_scores': data[['human_label', 'score', 'trained_score', 'acquisition']],
            'visualisation': visualisation, 
            'active_learning': gp_learning}

run_pipeline()