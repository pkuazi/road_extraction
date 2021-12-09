# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 10:39:08 2020

@author: ink
"""

# DBSCAN
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline 
# #############################################################################
# Generate sample data
df = pd.read_csv('E:/tmp/potsdam_2_10_indices.csv')
# #############################################################################
# Compute DBSCAN
# X=df.iloc[:, 1:65] 
X=df.iloc[:, 1:15] 
new_X=X.fillna(0)
pca=PCA(n_components=3, copy=True, whiten=False)
new_X = pca.fit_transform(new_X)

labels_true =df.iloc[:, -1] 
labels_true=labels_true.fillna(0)
new_X = StandardScaler().fit_transform(new_X)
for eps in np.arange(0.1,15,0.1):
    for min_samples in np.arange(2,10,1):
        db=DBSCAN(eps=eps, min_samples=min_samples).fit(new_X)
        labels = db.labels_
        v_measure =  metrics.v_measure_score(labels_true, labels)
        # Completeness = metrics.completeness_score(labels_true, labels)
        if v_measure<0.3:
            break
        else:
            print(eps, min_samples,v_measure)
           
        
db = DBSCAN(eps=1.6, min_samples=2).fit(new_X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

df['dbscan_label'] = labels
plt.figure(figsize=(12, 8))
sns.scatterplot(x=new_X[:, 0], y=new_X[:, 2], hue=labels_true, 
                palette=sns.color_palette('hls', np.unique(labels_true).shape[0]))
plt.title('DBSCAN with epsilon 11, min samples 6')
plt.show()

print('Estimated number of clusters: %d' % n_clusters_)

print('Estimated number of noise points: %d' % n_noise_)
# df.to_csv('E:/tmp/test_dbscan.csv', index=True)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(new_X, labels))

# #############################################################################
# Plot result
# import matplotlib.pyplot as plt

# # Black removed and is used for noise instead.
# unique_labels = set(labels)
# colors = [plt.cm.Spectral(each)
#           for each in np.linspace(0, 1, len(unique_labels))]
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = [0, 0, 0, 1]

#     class_member_mask = (labels == k)

#     xy = X[class_member_mask & core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=14)

#     xy = X[class_member_mask & ~core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=6)

# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()
# spectral clustering