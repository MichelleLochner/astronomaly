# An example with a subset of Galaxy Zoo data
from astronomaly.data_management import image_reader
from astronomaly.data_management import light_curve_reader
from astronomaly.preprocessing import image_preprocessing
from astronomaly.feature_extraction import shape_features
from astronomaly.feature_extraction import feets_features
from astronomaly.postprocessing import scaling
from astronomaly.anomaly_detection import isolation_forest, human_loop_learning
from astronomaly.visualisation import tsne
import os
import pandas as pd
import zipfile

# Root directory for data
data_dir = os.path.join(os.getcwd(), 'example_data')
lc_path = os.path.join(data_dir, 'transient_lightcurves.csv')

# Where output should be stored
output_dir = os.path.join(
    data_dir, 'astronomaly_output', '')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)



# These are transform functions that will be applied to images before feature
# extraction is performed. Functions are called in order.

# You can apply a different set of transforms to the images that get displayed
# in the frontend. In this case, I want to see the original images before sigma
# clipping is applied.
display_transform_function = []


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
    lc_dataset = light_curve_reader.LightCurveDataset(
        filename=lc_path, 
        data_dict = {'id':0,'time':4,'mag':2,'mag_err':3}
    )

    lc_dataset = lc_dataset.light_curves_data

    # Creates a pipeline object for feature extraction
    pipeline_feets = feets_features.Feets_Features(
        exclude_features=['Period_fit'])
    
    # Actually runs the feature extraction
    features = pipeline_feets.run_on_dataset(lc_dataset)

    # Now we rescale the features using the same procedure of first creating
    # the pipeline object, then running it on the feature set
    pipeline_scaler = scaling.FeatureScaler(force_rerun=False,
                                            output_dir=output_dir)
    features = pipeline_scaler.run(features)

    # The actual anomaly detection is called in the same way by creating an
    # Iforest pipeline object then running it
    pipeline_iforest = isolation_forest.IforestAlgorithm(
        force_rerun=False, output_dir=output_dir)
    anomalies = pipeline_iforest.run(features)

    # We convert the scores onto a range of 0-5
    pipeline_score_converter = human_loop_learning.ScoreConverter(
        force_rerun=False, output_dir=output_dir)
    anomalies = pipeline_score_converter.run(anomalies)

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
        pass

    # This is the active learning object that will be run on demand by the
    # frontend 
    pipeline_active_learning = human_loop_learning.NeighbourScore(
        alpha=1, output_dir=output_dir)

    # We use TSNE for visualisation which is run in the same way as other parts
    # of the pipeline.
    pipeline_tsne = tsne.TSNE_Plot(
        force_rerun=False,
        output_dir=output_dir,
        perplexity=100)
    t_plot = pipeline_tsne.run(features)

    # The run_pipeline function must return a dictionary with these keywords
    return {'dataset': lc_dataset, 
            'features': features, 
            'anomaly_scores': anomalies,
            'visualisation': t_plot, 
            'active_learning': pipeline_active_learning}