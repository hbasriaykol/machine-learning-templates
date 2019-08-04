#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: hbasriaykol

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datas = pd.read_csv('customers.csv')

x=datas.iloc[:,3:]


#KMeans Clustering
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters = 3, init = 'k-means++')
kmeans.fit(x)

print (kmeans.cluster_centers_)

'''WCSS'''
results = []
for i in range(1,10):
    kmeans = KMeans (n_clusters = i , init='k-means++', random_state=123)
    kmeans.fit(x)
    results.append(kmeans.inertia_)

plt.plot(range(1,10),results)


#Hierarchic Clustering - Agglomerative
from sklearn.cluster import AgglomerativeClustering
ac= AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage = 'ward')
y_pred = ac.fit_predict(x)
print(y_pred)