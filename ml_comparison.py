import os
import glob
import pickle
import random

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('notebook')
import pandas as pd
import numpy as np
from PIL import Image
from scipy import stats
import tqdm
from sklearn.metrics import recall_score

from sklearn.model_selection import train_test_split
from sklearn.decomposition import IncrementalPCA

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Matern

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from modAL.models import ActiveLearner, BayesianOptimizer
from modAL.acquisition import max_EI


def load_data():
    unshuffled_features, unshuffled_labels = load_raw_data()

    shuffle_indices= np.arange(len(unshuffled_labels))
    random.shuffle(shuffle_indices)  # inplace
    features = unshuffled_features[shuffle_indices]
    labels = unshuffled_labels[shuffle_indices]

    return features, labels


def load_raw_data():
    labels = np.load('example_data/Simulations/labels_test.npy').reshape(-1, 1)
    features = np.load('example_data/Simulations/y_test.npy')
    assert len(features) == len(labels)
    return features, labels


def get_embed(features, n_components, save=''):
    embedder = IncrementalPCA(n_components=n_components)
     # no train/test needed as unsupervised
    if len(save) > 0:
        plt.plot(embedder.explained_variance_)  # 5 would probably do?
        plt.savefig(save)
        plt.close()
    return embedder.fit_transform(features) 


def split_three_ways(data, labels, train_size, pool_test_frac=.2, random_state=0):
    # big pool/test
    # TODO wrap with crossval
    X_train, X_not, y_train, y_not = train_test_split(data, labels, train_size=train_size, random_state=random_state)
    # 80% pool, 20% test
    X_pool, X_test, y_pool, y_test = train_test_split(X_not, y_not, test_size=pool_test_frac)
    return X_train, y_train, X_pool, y_pool, X_test, y_test


def visualise_predictions(data, optimizer, save):
    preds = optimizer.predict(data)
    plt.scatter(data[:, 0], data[:, 1], c=np.squeeze(preds))
    plt.colorbar()
    plt.savefig(save)
    plt.close()


def teach_optimizer(optimizer, X_pool, y_pool, retrain_size):
    query_indices, query_instances = optimizer.query(X_pool, n_instances=retrain_size)
    optimizer.teach(X_pool[query_indices], y_pool[query_indices])


def get_metrics(optimizer, X_test, y_test):
    preds = optimizer.predict(X_test)
    
    sort_indices = np.argsort(np.squeeze(preds))[::-1]
    sorted_preds = preds[sort_indices]
    sorted_labels = y_test[sort_indices].squeeze()

    is_interesting = np.isclose(sorted_labels, 4)  # TODO get rws for all classes

    # print(is_interesting[:top_n])
    metrics = {}
    for top_n in [50, 150]:
        metrics.update({
            'total_found_{}'.format(top_n): is_interesting[:top_n].sum(),
            'recall_{}'.format(top_n): get_recall(is_interesting, top_n),  # using top_n as set
            'rank_weighted_score_{}'.format(top_n): get_rank_weighted_score(is_interesting, n=top_n)
        })
    return metrics
    

def get_recall(is_interesting, top_n):
    consider_anomaly = np.concatenate([np.ones(top_n), np.zeros(len(is_interesting) - top_n)])
    return recall_score(is_interesting, consider_anomaly)
    #  could do manually simply with is_interesting[:top_n].sum() / is_interesting.sum()


def get_rank_weighted_score(is_interesting, n):
    # astronomaly eqn 4
    # n = "number of objects a human may reasonably look at"
    # is_interesting is the indicator: true if label in interesting class, false otherwise

    is_interesting_top = is_interesting[:n]

    i = np.arange(0, n)
    weights = n - i  # start from index=0 and lose the +1, more conventional
    # and put the -1 here in the normalisation to compensate
    s_zero = (n * (n+1) / 2)  # sum of integers from 0 to N (i.e. the weights * indicator if all are anomalies, max possible)
    return np.sum(weights * is_interesting_top) / s_zero


# def check_if_interesting(labels, anomaly_labels=[4]):
#     interesting = np.zeros_like(labels).astype(bool)  # i.e. False everywhere
#     for anomaly_label in anomaly_labels:
#         interesting = interesting | labels.astype(int) == anomaly_label
#     return interesting


if __name__ == '__main__':

    features, labels = load_data()
    embed = get_embed(features, n_components=5)

    # embed_subset, labels_subset = embed[:5000], labels[:5000]
    # sns.scatterplot(x=embed_subset[:, 0], y=embed_subset[:, 1], hue=np.squeeze(labels_subset), alpha=.3)
    # plt.savefig('simulated_embed_first_2_components.png')
    # plt.close()
    
    all_metrics = []
    for random_state in np.arange(20):

        X_train, y_train, X_pool, y_pool, X_test, y_test = split_three_ways(embed, labels, train_size=10, random_state=0)
        # print(X_train.shape, X_pool.shape, X_test.shape)
        # print(y_train.shape, y_pool.shape, y_test.shape)
        print('Total interesting: {}'.format(np.isclose(y_test, 4).sum()))

        kernel = RBF() + WhiteKernel()  # or matern
        gp = GaussianProcessRegressor(kernel=kernel, random_state=0)

        optimizer = BayesianOptimizer(
            estimator=gp,
            query_strategy=max_EI,
            X_training=X_train,
            y_training=y_train
        )

        retrain_size = 10
        retrain_batches = 9
        for retrain_batch in range(retrain_batches):
            labelled_samples = len(X_train) + (retrain_batch+1) * retrain_size
            # print('Labelled samples: {}'.format(labelled_samples))
            teach_optimizer(optimizer, X_pool, y_pool, retrain_size)
            metrics = get_metrics(optimizer, X_test, y_test)
            metrics['labelled_samples'] = labelled_samples
            all_metrics.append(metrics)

    df = pd.DataFrame(data=all_metrics)

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 8), sharex=True)

    sns.lineplot(x='labelled_samples', y='total_found_50', data=df, label='N=50', ax=ax0)
    sns.lineplot(x='labelled_samples', y='total_found_150', data=df, label='N=150', ax=ax0)
    ax0.set_ylim([0., None])
    ax0.set_ylabel('Anomalies Found')

    sns.lineplot(x='labelled_samples', y='recall_50', data=df, label='N=50', ax=ax1)
    sns.lineplot(x='labelled_samples', y='recall_150', data=df, label='N=150', ax=ax1)
    ax1.set_ylim([0., 1.])
    ax1.set_ylabel('Recall')

    sns.lineplot(x='labelled_samples', y='rank_weighted_score_50', data=df, label='N=50', ax=ax2)
    sns.lineplot(x='labelled_samples', y='rank_weighted_score_150', data=df, label='N=150', ax=ax2)
    ax2.set_xlabel('Labelled Examples')
    ax2.set_ylabel('Rank Weighted Score')
    ax2.set_ylim([0., None])

    fig.tight_layout()
    fig.savefig('ml_comparison_gp_metrics.png')
