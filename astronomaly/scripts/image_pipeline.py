from astronomaly.data_management import image_reader
from astronomaly.preprocessing import image_preprocessing
from astronomaly.feature_extraction import power_spectrum
from astronomaly.dimensionality_reduction import decomposition
from astronomaly.postprocessing import scaling
from astronomaly.anomaly_detection import isolation_forest, human_loop_learning
from astronomaly.clustering import tsne
import os


image_dir = '/home/michelle/BigData/Anomaly/GOODS_S/'
output_dir = '/home/michelle/BigData/Anomaly/astronomaly_output/images/'


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

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if len(image_dir) != 0:
        image_dataset = image_reader.ImageDataset(
            image_dir,
            transform_function=image_preprocessing.image_transform_log,
            window_size=128)

        pipeline_psd = power_spectrum.PSD_Features(force_rerun=False, 
                                                   output_dir=output_dir)
        features_original = pipeline_psd.run_on_dataset(image_dataset)

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

        pipeline_active_learning = human_loop_learning.NeighbourScore(alpha=1)

        pipeline_tsne = tsne.TSNE_Plot(force_rerun=False,
                                       output_dir=output_dir, 
                                       perplexity=50)
        t_plot = pipeline_tsne.run(features)
        # t_plot = np.log(features_scaled + np.abs(features_scaled.min())+0.1)

        return {'dataset': image_dataset, 
                'features': features, 
                'anomaly_scores': anomalies,
                'cluster': t_plot, 
                'active_learning': pipeline_active_learning}
    else:
        return None


# run_pipeline(image_dir='/home/michelle/BigData/Anomaly/Meerkat_deep2/')
