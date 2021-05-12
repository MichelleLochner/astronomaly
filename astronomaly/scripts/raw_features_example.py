# Replicates the simulated example in the paper
import os
import numpy as np
from astronomaly.data_management import raw_features
from astronomaly.anomaly_detection import lof, human_loop_learning
from astronomaly.visualisation import tsne

# Root directory for data
data_dir = os.path.join(os.getcwd(), 'example_data', )

input_files = [os.path.join(data_dir, 'Simulations', 'y_test.npy'), 
               os.path.join(data_dir, 'Simulations', 'labels_test.npy')]

# Where output should be stored
output_dir = os.path.join(
    data_dir, 'astronomaly_output', 'simulations', '')


def artificial_human_labelling(anomalies=None, metadata=None, N=200, 
                               human_labels={0: 0, 1: 0, 2: 3, 3: 0, 4: 5}):

    print('Artificially adding human labels...')
    if anomalies is None:
        raise ValueError('Anomaly score dataframe not provided')
    if metadata is None:
        raise ValueError('True labels not given')

    anomalies['human_label'] = [-1] * len(anomalies)

    labels = metadata.loc[anomalies.index]
    for k in list(human_labels.keys()):
        inds = labels.index[:N][(np.where(labels.label[:N] == k))[0]]
        anomalies.loc[inds, 'human_label'] = human_labels[k]

    print('Done!')

    return anomalies


def run_pipeline():

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    raw_dataset = raw_features.RawFeatures(list_of_files=input_files, 
                                           output_dir=output_dir)
    features = raw_dataset.features

    pipeline_lof = lof.LOF_Algorithm(output_dir=output_dir, n_neighbors=100, 
                                     force_rerun=False)
    anomalies = pipeline_lof.run(features)

    pipeline_score_converter = human_loop_learning.ScoreConverter(
        output_dir=output_dir)
    anomalies = pipeline_score_converter.run(anomalies)
    anomalies = anomalies.sort_values('score', ascending=False)

    anomalies = artificial_human_labelling(
        anomalies=anomalies, metadata=raw_dataset.metadata, N=200, 
        human_labels={0: 0, 1: 0, 2: 3, 3: 0, 4: 5})

    pipeline_active_learning = human_loop_learning.NeighbourScore(
        alpha=1, force_rerun=True, output_dir=output_dir)

    pipeline_tsne = tsne.TSNE_Plot(output_dir=output_dir, perplexity=50)
    t_plot = pipeline_tsne.run(features.loc[anomalies.index])

    return {'dataset': raw_dataset, 
            'features': features, 
            'anomaly_scores': anomalies,
            'visualisation': t_plot, 
            'active_learning': pipeline_active_learning}
