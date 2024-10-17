# An example with a subset of Galaxy Zoo data using BYOL features and Protege
# Performance is not great with only 1000 training examples but this can be
# used as a template for other applications.
from astronomaly.data_management import image_reader
from astronomaly.preprocessing import image_preprocessing
from astronomaly.feature_extraction import byol_features
from astronomaly.postprocessing import scaling
from astronomaly.anomaly_detection import isolation_forest, human_loop_learning
from astronomaly.anomaly_detection import protege
from astronomaly.dimensionality_reduction import pca
from astronomaly.visualisation import umap_plot
from astronomaly.utils.utils import pca_based_initial_selection
import os
import pandas as pd
import zipfile

# Root directory for data
data_dir = os.path.join(os.getcwd(), 'example_data')

image_dir = os.path.join(data_dir, 'GalaxyZooSubset', '')

# Where output should be stored
output_dir = os.path.join(
    data_dir, 'astronomaly_output', 'galaxy_zoo_byol_protege', '')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(image_dir):
    # Data has not been unzipped yet
    zip_ref = zipfile.ZipFile(os.path.join(data_dir, 'GalaxyZooSubset.zip'))
    zip_ref.extractall(data_dir)


# These are transform functions that will be applied to images before feature
# extraction is performed. Functions are called in order.
image_transform_function = [
    image_preprocessing.image_transform_greyscale,
    image_preprocessing.image_transform_sigma_clipping,
    image_preprocessing.image_transform_remove_negatives,
    image_preprocessing.image_transform_scale]

# You can apply a different set of transforms to the images that get displayed
# in the frontend. In this case, I want to see the original images before sigma
# clipping is applied.
display_transform_function = [
    image_preprocessing.image_transform_scale]

force_rerun = False
# Either 'pca' or 'random' for initial sample for protege
initial_sorting = 'pca' 

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
    image_dataset = image_reader.ImageThumbnailsDataset(
        directory=image_dir, output_dir=output_dir, 
        transform_function=image_transform_function,
        display_transform_function=display_transform_function,
        display_image_size=424
    )

    if force_rerun or not os.path.exists(output_dir + 'byol_model.pt'):
        do_training = True
        load_model = False
    else:
        do_training = False
        load_model = True

    # Creates a pipeline object for feature extraction
    pipeline_byol = byol_features.BYOL_Features(
        output_dir=output_dir, 
        image_size=128,
        n_epochs=100,
        batch_size=32, # Larger is usually better, 32 recommended
        num_workers=16, # Roughly the number of CPUs you have
        load_model=load_model,
        force_rerun=force_rerun)

    if do_training:
        pipeline_byol.train_byol(image_dataset)

    # Actually runs the feature extraction
    features_original = pipeline_byol.run_on_dataset(image_dataset)

    pipeline_pca = pca.PCA_Decomposer(
        force_rerun=force_rerun, output_dir=output_dir, threshold=0.85)
    features = pipeline_pca.run(features_original)
    print('Features shape', features.shape)


    # Get initial sample of sources for Protege to ask the user for scoring
    anomalies = pca_based_initial_selection(features, 10)
    if initial_sorting == 'random':
        anomalies['score'] = 0
        rand_inds = np.random.choice(anomalies.index, size=10, replace=False)
        anomalies.loc[rand_inds, 'score'] = 5


    # # Protege
    pipeline_active_learning = protege.GaussianProcess(
        features, output_dir=output_dir, force_rerun=force_rerun, ei_tradeoff=3
    )

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

    

    # We use UMAP for visualisation which is run in the same way as other parts
    # of the pipeline.
    pipeline_umap = umap_plot.UMAP_Plot(
        force_rerun=False,
        output_dir=output_dir)
    vis_plot = pipeline_umap.run(features)

    # The run_pipeline function must return a dictionary with these keywords
    return {'dataset': image_dataset, 
            'features': features, 
            'anomaly_scores': anomalies,
            'visualisation': vis_plot, 
            'active_learning': pipeline_active_learning}
