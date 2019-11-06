from astronomaly.data_management import light_curve_reader
from astronomaly.postprocessing import scaling
from astronomaly.anomaly_detection import isolation_forest, human_loop_learning
from astronomaly.clustering import tsne
import os
import pandas as pd


input_dir = '/home/michelle/BigData/Anomaly/dwf_data/'
output_dir = '/home/michelle/BigData/Anomaly/astronomaly_output/light_curves/'


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

    if len(input_dir) != 0:
        light_curve_dataset = light_curve_reader.LightCurveDataset(
            directory=os.path.join(input_dir, 'light_curves'))

        flpath = os.path.join(input_dir, 
                              'feature_table_large_mag_zeros_remove.ascii')
        features = pd.read_csv(flpath, delim_whitespace=True)
        features = features.set_index('LC_name')

        pipeline_scaler = scaling.FeatureScaler(
            force_rerun=False, output_dir=output_dir)
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

        return {'dataset': light_curve_dataset, 
                'features': features, 
                'anomaly_scores': anomalies,
                'cluster': t_plot, 
                'active_learning': pipeline_active_learning}
    else:
        return None
