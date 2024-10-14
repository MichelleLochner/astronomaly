# An example for the CRTS data
from astronomaly.data_management import light_curve_reader
from astronomaly.postprocessing import scaling
from astronomaly.anomaly_detection import isolation_forest, human_loop_learning
from astronomaly.visualisation import umap_plot
import os
import pandas as pd

try:
    from astronomaly.feature_extraction import feets_features
    import_failed = False
except ImportError:
    print("""WARNING: Failed to import feets features.
    Feets is no longer maintained and not compatible with the latest 
    version of Astropy. Install astropy v5.3.4 in a virtual environment to 
    run. Previously extracted features will now be read from file.""")
    import_failed = True


# Root directory for data
data_dir = os.path.join(os.getcwd(), 'example_data')
lc_path = os.path.join(data_dir, 'CRTS', 'CRTS_subset_500.csv')

# Where output should be stored
output_dir = os.path.join(
    data_dir, 'astronomaly_output', 'CRTS', '')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

display_transform_function = []
# Change this to false to automatically use previously run features
force_rerun = True 


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
        data_dict={'id': 0, 'time': 4, 'mag': 2, 'mag_err': 3},
        output_dir=output_dir
    )

    if not import_failed:
        # Creates a pipeline object for feature extraction
        pipeline_feets = feets_features.Feets_Features(
            exclude_features=['Period_fit', 'PercentDifferenceFluxPercentile',
                            'FluxPercentileRatioMid20',
                            'FluxPercentileRatioMid35',
                            'FluxPercentileRatioMid50',
                            'FluxPercentileRatioMid65',
                            'FluxPercentileRatioMid80'],
            compute_on_mags=True,
            # Feets prints a lot of warnings to screen, set this to true to ignore
            # You may also want to run with `python -W ignore` (with caution)
            ignore_warnings=True,  
            output_dir=output_dir,
            force_rerun=force_rerun)

        # Actually runs the feature extraction
        features = pipeline_feets.run_on_dataset(lc_dataset)
    else:
        print("Reading features from file...")
        features = pd.read_parquet(os.path.join(
            data_dir, 'CRTS', 'Feets_Features_output.parquet'))

    # Now we rescale the features using the same procedure of first creating
    # the pipeline object, then running it on the feature set
    pipeline_scaler = scaling.FeatureScaler(force_rerun=force_rerun,
                                            output_dir=output_dir)
    features = pipeline_scaler.run(features)

    # The actual anomaly detection is called in the same way by creating an
    # Iforest pipeline object then running it
    pipeline_iforest = isolation_forest.IforestAlgorithm(
        force_rerun=force_rerun, output_dir=output_dir)
    anomalies = pipeline_iforest.run(features)

    # We convert the scores onto a range of 0-5
    pipeline_score_converter = human_loop_learning.ScoreConverter(
        force_rerun=force_rerun, output_dir=output_dir)
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

    # We use UMAP for visualisation which is run in the same way as other parts
    # of the pipeline.
    pipeline_umap = umap_plot.UMAP_Plot(
        force_rerun=force_rerun,
        output_dir=output_dir)
    vis_plot = pipeline_umap.run(features)

    # The run_pipeline function must return a dictionary with these keywords
    return {'dataset': lc_dataset,
            'features': features,
            'anomaly_scores': anomalies,
            'visualisation': vis_plot,
            'active_learning': pipeline_active_learning}
