import os
from astronomaly.data_management import raw_features
from astronomaly.preprocessing import scaling
from astronomaly.anomaly_detection import lof, human_loop_learning
from astronomaly.clustering import tsne

input_file = \
    '/home/michelle/Project/OutlierDetection/outlier_detection/data/y_test.npy'
output_dir = '/home/michelle/BigData/Anomaly/astronomaly_output/badac/'


def run_pipeline():

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    raw_dataset = raw_features.RawFeatures(filename=input_file, 
                                           output_dir=output_dir)
    features = raw_dataset.features

    pipeline_scaler = scaling.FeatureScaler(output_dir=output_dir)
    features = pipeline_scaler.run(raw_dataset.features)

    pipeline_lof = lof.LOF_Algorithm(output_dir=output_dir)
    anomalies = pipeline_lof.run(features)

    pipeline_score_converter = human_loop_learning.ScoreConverter(
        output_dir=output_dir)
    anomalies = pipeline_score_converter.run(anomalies)
    anomalies = anomalies.sort_values('score', ascending=False)

    pipeline_active_learning = human_loop_learning.NeighbourScore(
        alpha=1, force_rerun=True, output_dir=output_dir)

    pipeline_tsne = tsne.TSNE_Plot(output_dir=output_dir, perplexity=50)
    t_plot = pipeline_tsne.run(features)

    return {'dataset': raw_dataset, 
            'features': features, 
            'anomaly_scores': anomalies,
            'cluster': t_plot, 
            'active_learning': pipeline_active_learning}
