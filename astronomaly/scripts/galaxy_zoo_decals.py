import os
import pandas as pd
import numpy as np
import zipfile
import json
from PIL import Image
import glob

os.environ['FLASK_ENV'] = 'development'

from astronomaly.data_management import image_reader
from astronomaly.preprocessing import image_preprocessing
from astronomaly.feature_extraction import shape_features
from astronomaly.postprocessing import scaling
from astronomaly.anomaly_detection import isolation_forest, human_loop_learning
from astronomaly.visualisation import tsne

# Where output should be stored
output_dir = os.path.join('/home/walml/repos/astronomaly/temp')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Where output should be stored
# output_dir = os.path.join(
#     data_dir, 'astronomaly_output', '')
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)


image_transform_function = None
display_transform_function = None

max_galaxies = 1000
feature_loc = 'dr5_8_b0_pca2_and_safe_ids.parquet'  # 2D
# feature_loc = 'dr5_8_b0_pca10_and_safe_ids.parquet'  # 10D
feature_df = pd.read_parquet(feature_loc)
feature_df = feature_df[feature_df['galaxy_id'].str.startswith('J')][:max_galaxies]

feature_cols = [col for col in feature_df.columns.values if col.startswith('feat')]
features = feature_df[feature_cols + ['galaxy_id']]  # already pca'd
features = features.set_index('galaxy_id')  # features must be string-indexed by objid, is not set to index internally, need to do manually
features.index = features.index.rename('objid') # probably not needed

full_catalog = pd.read_parquet('dr5_dr8_catalog_with_radius.parquet')
catalog = pd.merge(full_catalog, feature_df, on='galaxy_id', how='inner').reset_index(drop=True)
catalog['objid'] = catalog['galaxy_id'].astype(str)  # objid is set as index internally (but only after assign bug fixed)
catalog['filename'] = catalog['png_loc']


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

    # The actual anomaly detection is called in the same way by creating an
    # Iforest pipeline object then running it
    pipeline_iforest = isolation_forest.IforestAlgorithm(
        force_rerun=False, output_dir=output_dir)
    anomalies = pipeline_iforest.run(features)
    # returns dataframe with "score" column and features.index as index

    # We convert the scores onto a range of 0-5
    pipeline_score_converter = human_loop_learning.ScoreConverter(
        force_rerun=False, output_dir=output_dir)
    anomalies = pipeline_score_converter.run(anomalies)

    # anomalies df will now be modified to add human_label column if there are previous human_labels, saved (confusingly) in ml_scores.csv 
    # (because it's added as a column to anomalies df rather than its own csv...)
    try:
        # This is used by the frontend to store labels as they are applied so
        # that labels are not forgotten between sessions of using Astronomaly
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
        pass  # will save ml_scores.csv every time controller.get_label is called

    # This is the active learning object that will be run on demand by the
    # frontend 
    pipeline_active_learning = human_loop_learning.NeighbourScore(
        alpha=1, output_dir=output_dir)

    # We use TSNE for visualisation which is run in the same way as other parts
    # of the pipeline.
    # pipeline_tsne = tsne.TSNE_Plot(
    #     force_rerun=False,
    #     output_dir=output_dir,
    #     perplexity=100)
    # t_plot = pipeline_tsne.run(features)
    t_plot = None

    # The run_pipeline function must return a dictionary with these keywords
    return {'dataset': image_dataset, 
            'features': features, 
            'anomaly_scores': anomalies,
            'visualisation': t_plot, 
            'active_learning': pipeline_active_learning}

# run_pipeline()