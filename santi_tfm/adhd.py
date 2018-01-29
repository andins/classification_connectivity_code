#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 09:10:40 2018

@author: andrea
"""

import numpy as np
import matplotlib.pyplot as plt
from nilearn.datasets import fetch_adhd, fetch_atlas_aal
from nilearn.image import mean_img, load_img, index_img
from nilearn.plotting import plot_epi, plot_stat_map, plot_roi, plot_matrix
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure

adhd = fetch_adhd(n_subjects=3, data_dir='/home/andrea/Work/santi_tfm')
mean = mean_img(adhd.func[0])
plot_epi(mean)
img = load_img(adhd.func[0])
first_vol = index_img(adhd.func[0], 0)
plot_stat_map(first_vol)

# %%
atlas = fetch_atlas_aal()
maps = atlas.maps
labels = atlas.labels
# plot_roi(maps)  # plot parcels
masker = NiftiLabelsMasker(labels_img=maps, standardize=True)
time_series = masker.fit_transform(adhd.func[0], confounds=adhd.confounds)
correl = ConnectivityMeasure(kind='correlation')
FC = correl.fit_transform([time_series])[0]
np.fill_diagonal(FC, 0)
plot_matrix(FC, figure=(10, 8), labels=labels[1:],
            vmax=0.8, vmin=-0.8)
