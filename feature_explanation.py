import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import defaultdict

def get_state(doc, i):
    return (doc[i], i//64)

def train_model(doc_ids_train, target_feature_train):
    means = defaultdict(list)
    for doc, feature_values in zip(doc_ids_train, target_feature_train):
        for i in range(len(doc)):
            state = get_state(doc, i)
            means[state].append(feature_values[i])
    stats = {k: sum(v)/len(v) for k, v in means.items()}
    train_mean = target_feature_train.mean()
    return lambda doc, idx: stats.get(get_state(doc, idx), train_mean)

def get_mse_ratio(model, train_mean, doc_ids_test, target_feature_test):
    model_predictions = np.array([[model(doc_ids_test[i, :], j) for j in range(doc_ids_test.shape[1])] for i in range(doc_ids_test.shape[0])])
    mse = np.sum((target_feature_test - model_predictions) ** 2)
    baseline_mse = np.sum((target_feature_test - train_mean) ** 2)
    return 1 - mse/baseline_mse

def create_histogram_plot(n_features):
    activations = torch.load('data/mlp_F6000_clipped_sparse.pt')
    doc_ids = np.array(torch.load('data/train_tok_ids.pt'))
    feature_ids = list(range(6000))
    random.shuffle(feature_ids)

    mse_ratios = []

    for feature_id in tqdm(feature_ids[:n_features]):
        target_feature = np.array(activations[feature_id].to_dense())
        target_feature = target_feature / target_feature.max()
        doc_ids_train, doc_ids_test, target_feature_train, target_feature_test = train_test_split(
            doc_ids, target_feature, test_size=0.5, random_state=42)
        model = train_model(doc_ids_train, target_feature_train)
        train_mean = target_feature_train.mean()
        mse_ratio = get_mse_ratio(model, train_mean, doc_ids_test, target_feature_test)
        mse_ratios.append(mse_ratio * 100)

    # Plotting the histogram
    plt.hist(mse_ratios, bins=30, alpha=0.7, color='grey')
    plt.xlabel('% of MSE Explained')
    plt.ylabel('Frequency')
    plt.title(f'% of MSE Explained Across {n_features} Random Features')
    plt.xlim(0, 100)
    
    # Calculate and annotate the mean
    mean_ratio = np.mean(mse_ratios)
    plt.axvline(mean_ratio, color='black', linestyle='dashed', linewidth=1)
    plt.text(mean_ratio, plt.ylim()[1]*0.9, f' Mean: {mean_ratio:.2f} %', color='black')
    plt.savefig('explained_mse_plot.svg', format='svg')
