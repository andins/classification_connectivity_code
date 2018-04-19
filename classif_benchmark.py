#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 15:50:53 2018

@author: andrea
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from MOU import MOU


def make_timeseries_classification(N=50, density=0.2, n_classes=2, n_samples=20, T=150):
    """
    PARAMETERS:
        N : number of nodes in the network
        density : average density of the connectivity matrix
        n_classes : number of classes (one adjacency matrix generated for each)
        n_samples : list-like of shape [n_classes] or int, number of samples for each class (if scalar create balanced classes)
        T : number of time points
    RETURNS:
        ts : shape [T, N, np.sum(n_samples)], time series
        y : shape[np.sum(n_samples)], class labels
    """
    if np.isscalar(n_samples):  # if scalar create balanced dataset
        n_samples = np.ones([n_classes], dtype=int) * n_samples
    elif len(n_samples) != n_classes:
        raise ValueError("n_samples should be scalar or its length has to be equal to n_classes!")
    # for each cond generate a binary adjacency matrix A
    A = np.ones([N, N, n_classes])
    ts = np.zeros([T, N, np.sum(n_samples)])
    y = np.zeros([np.sum(n_samples)])
    for c in range(n_classes):
        A[np.random.rand(N, N) > density, c] = 0
        # for each sample generate a continuous weight matrix W
        for s in range(n_samples[c]):
            W = np.exp(np.random.randn(N, N))
            C = W * A[:, :, c]
            C *= 0.5 * N / C.sum()
            # generate time series with MOU.simulate(C=A*W)
            ts[:, :, s + int(np.sum(n_samples[:c]))] = MOU(n_nodes=N, C=C).simulate(T=T)
            y[s + int(np.sum(n_samples[:c]))] = c
    return ts, y


# generate time series
N = 50
density = 0.2
ts, y = make_timeseries_classification(N=N, density=density, n_classes=30, n_samples=5, T=300)

# calculate FC and EC and build data matrix X
X = np.zeros([ts.shape[2], int(N * (N-1) / 2)])
for s in range(ts.shape[2]):
    # estimate FC from time series
    FC = np.corrcoef(ts[:, :, s].T)
    X[s, :] = FC[np.tril_indices_from(FC, k=-1)]
    # estimate EC from time series

# predict cond from FC and EC

# classifier: logistic regression
clf = LogisticRegression(C=10000, penalty='l2', multi_class= 'multinomial', solver='lbfgs')

# corresponding pipeline: zscore and pca can be easily turned on or off
pipe_z_pca_mlr = Pipeline([('zscore', StandardScaler()), 
			 ('pca', PCA()),
                         ('clf', clf)])
repetitions = 100  # number of times the train/test split is repeated
# shuffle splits for validation test accuracy
shS = ShuffleSplit(n_splits=repetitions, test_size=None, train_size=.8, random_state=0)

score_z_pca_mlr = np.zeros([repetitions])
i = 0  # counter for repetitions
for train_idx, test_idx in shS.split(X):  # repetitions loop
    data_train = X[train_idx, :]
    y_train = y[train_idx]
    data_test = X[test_idx, :]
    y_test = y[test_idx]
    pipe_z_pca_mlr.fit(data_train, y_train)
    score_z_pca_mlr[i] = pipe_z_pca_mlr.score(data_test, y_test)
    i += 1
        
# plot comparison as violin plots
fig, ax = plt.subplots()
sns.violinplot(data=[score_z_pca_mlr], cut=0, orient='h', scale='width')
ax.set_yticklabels(['z+PCA+MLR'])