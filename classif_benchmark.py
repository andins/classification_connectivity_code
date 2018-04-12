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
# for each cond generate a binary adjacency matrix A
N = 50
density = 0.2
A_1 = np.ones([N, N])
A_1[np.random.rand(N, N) > density] = 0
A_2 = np.ones([N, N])
A_2[np.random.rand(N, N) > density] = 0

# for each sample generate a continuous weight matrix W
C_1 = np.zeros([N, N, 20])
C_2 = np.zeros([N, N, 20])
ts_1 = np.zeros([300, N, 20])
ts_2 = np.zeros([300, N, 20])
X = np.zeros([40, int(N * (N-1) / 2)])
for s in range(20):
    C_1[:, :, s] = np.exp(np.random.randn(N, N)) * A_1
    C_1[:, :, s] *= 0.5 * N / C_1[:, :, s].sum()
    C_2[:, :, s] = np.exp(np.random.randn(N, N)) * A_2
    C_2[:, :, s] *= 0.5 * N / C_2[:, :, s].sum()
    # generate time series with MOU.simulate(C=A*W)
    ts_1[:, :, s] = MOU(n_nodes=N, C=C_1[:, :, s]).simulate(T=300)
    ts_2[:, :, s] = MOU(n_nodes=N, C=C_2[:, :, s]).simulate(T=300)
    # estimate FC from time series
    FC = np.corrcoef(ts_1[:, :, s].T)
    fcflat = np.tril(FC, k=-1).flatten()
    X[s, :] = fcflat[np.flatnonzero(fcflat)]
    FC = np.corrcoef(ts_2[:, :, s].T)
    fcflat = np.tril(FC, k=-1).flatten()
    X[s + 20, :] = fcflat[np.flatnonzero(fcflat)]
    # estimate EC from time series

# targets
y = np.ones(40)
y[:20] = 0
np.random.shuffle(y)

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
    i+=1
        
# plot comparison as violin plots
fig, ax = plt.subplots()
sns.violinplot(data=[score_z_pca_mlr], cut=0, orient='h', scale='width')
ax.set_yticklabels(['z+PCA+MLR'])