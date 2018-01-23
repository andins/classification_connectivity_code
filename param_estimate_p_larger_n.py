#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 08:07:57 2018

@author: andrea
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import ShuffleSplit
import scipy.linalg as spl


# datasets with p>n parameters
D = 50  # number of datasets
N = 200  # number of samples
n_dim = 300  # dimensionality of datasets
# generate all datasets at ones (to be splitted later)
Xall = np.random.randn(N*D, n_dim)
yall = np.concatenate(([0]*int(N*D/2), [1]*int(N*D/2)))
#Xall, yall = make_classification(n_samples=N*D, n_features=n_dim,
#                               n_informative=1, n_redundant=0,
#                               n_repeated=0, class_sep=5.0,
#                               n_clusters_per_class=1, shuffle=False)
plt.figure()
plt.scatter(Xall[yall==0,0], Xall[yall==0,1])
plt.scatter(Xall[yall==1,0], Xall[yall==1,1])

rnd_idx = list(range(N*D))
np.random.shuffle(rnd_idx)
Xall = Xall[rnd_idx, :]
yall = yall[rnd_idx]
# H1: different disjoint sets of samples from the same ground truth produce
#     very different estimates of parameters dispite similar test accuracy
repetitions = 10  # repetitions to calculate test-set accuracy
param = np.zeros([D, n_dim])  # init (there are n_dim coefficients)
scores = np.zeros([D, repetitions])  # test set scores of classifiers
clf = dict()
# for each dataset
for d in range(D):
    X = Xall[N*d:N+N*d, :]
    y = yall[N*d:N+N*d]
    clf[d] = LinearRegression()
    rs = ShuffleSplit(n_splits=repetitions, test_size=0.8)
    r = 0
    # calculate test set accuracy ###############################
    for tr_idx, ts_idx in rs.split(X):
        clf[d].fit(X[tr_idx, :], y[tr_idx])
        scores[d, r] = clf[d].score(X[ts_idx, :], y[ts_idx])
        r += 1
    #############################################################
    param[d, :] = clf[d].coef_
# calculate variability of parameters as 1/D \sum_d ||\theta_d - \bar\theta||
diffs = param - np.repeat(np.reshape(param.mean(axis=0), [1, n_dim]), D, axis=0)
variability = 0
w1 = np.zeros([D])
w2 = np.zeros([D])
for d in range(D):
    variability += spl.norm(diffs[d, :]) / D
    w1[d] = clf[d].coef_[0]
    w2[d] = clf[d].coef_[1]
plt.figure()
plt.plot(range(D), scores.mean(axis=1))
plt.fill_between(range(D), scores.mean(axis=1) +
                 scores.std(axis=1), scores.mean(axis=1) -
                 scores.std(axis=1), alpha=0.5)
plt.figure()
plt.scatter(w1, w2)
plt.scatter(param.mean(axis=0)[0], param.mean(axis=0)[1], color=[1, 0, 0])
# H2: the high variability of H1 is due to infinite solutions: as the
#     number of datasets used to calculate the variability is increased
#     the variability increases without bound
