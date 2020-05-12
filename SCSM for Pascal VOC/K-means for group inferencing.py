# # Generate the disimilar matrix from Dataframe:
# CLASSES  = ['Aeroplane','Bicycle','Bird','Boat','Bottle','Bus','Car','Cat','Chair','Cow' ,'Diningtable' ,
#             'dog' ,'horse','motorbike' ,'person' ,'pottedplant' ,'sheep' ,'sofa' ,'train' ,'tvmonitor']
# import pandas as pd
# df = pd.read_csv('C:/Users/Siri/PycharmProjects/Dataset_conversion/correct voc 2007 matrix.csv')
# for i in range(0, len(CLASSES)):
#     for j in range(0, 20):
#         df[CLASSES[i]][j] = 1 - df[CLASSES[i]][j]
#
# print(df.head())
# print(df.head())
# df.to_csv('Correct_VOC_2007_ANOVA_disimilar_matrix.csv')


CLASSES  = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

#Draw kmeans clustering
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import numpy as np
import io, urllib
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial import distance

import pandas as pd
df = pd.read_csv('correct voc 2007 matrix_kmeans.csv')
X = df.values.tolist()
output = SpectralClustering(3).fit_predict(X)
print(output)
# print(X)
# print(len(X))

kmeans = KMeans(n_clusters=3, random_state=0).fit(df)

first_cluster = []
second_cluster =[]
third_cluster =[]

kmeans.labels_ = output
print(kmeans.labels_)
print(len(kmeans.labels_))
for i in range(0, len(CLASSES)):
    if kmeans.labels_[i] == 0:
        first_cluster.append(CLASSES[i])
    elif kmeans.labels_[i] ==1:
        second_cluster.append(CLASSES[i])
    elif kmeans.labels_[i] ==2:
        third_cluster.append(CLASSES[i])

print(first_cluster)
print(second_cluster)
print(third_cluster)
print(kmeans.labels_)

