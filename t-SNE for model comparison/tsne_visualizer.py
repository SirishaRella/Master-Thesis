import math

import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats as stats
CLASSES = ['aeroplane','bicycle','bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


df = pd.read_csv('C:/Users/Siri/PycharmProjects/Dataset_conversion/Pascal_VOC/classes_2007/tvmonitor/metadata_features_tsne_backup.tsv',delimiter='\t', encoding='utf-8')
df = df.iloc[0:, 1:4]
# df1 = pd.read_csv('C:/Users/Siri/PycharmProjects/Dataset_conversion/Pascal_VOC/classes_2007/bicycle/metadata_features_tsne.tsv',delimiter='\t', encoding='utf-8')
# df['label'] = 1
class_list =[]
# df1['label'] = 0
for i in range(0, len(CLASSES)):
    print(CLASSES[i])
    df1 =pd.read_csv("C:/Users/Siri/PycharmProjects/Dataset_conversion/Pascal_VOC/classes_2007/"+CLASSES[i]+"/metadata_features_tsne_backup.tsv",delimiter='\t',encoding='utf-8')
    df1 = df1.iloc[0:, 1:4]

    # print(df)
    from scipy import stats

    # import numpy as np
    #
    # np.random.seed(12345678)
    temp = df.mean(axis=0)
    temp1 = df1.mean(axis=0)
    temps1 = [temp['x'], temp['y'], temp['z']]
    temps2 = [temp1['x'], temp1['y'], temp1['z']]
    res = stats.f_oneway(temps1, temps2)
    print(res[1])
    class_list.append(round(res[1],2))
print(class_list)

# print(df.head())
# X = df.iloc[1:,1:-1]
# Y= df.iloc[1:,-1]
#
# X1 = df1.iloc[1:,1:-1]
#
# Y1= df1.iloc[:,-1]



#
# labels = Y
# plt.figure(figsize=(116,150))
# plt.title("House Vs Apartment")
# plt.scatter(df.iloc[:,1],df.iloc[:,2], c='b', alpha=0.5, label ='house')
# plt.scatter(df1.iloc[:,1],df1.iloc[:,2], c='r', alpha =0.5, label='apartment')
# plt.legend(loc='upper left')
# plt.show()

#Concat two dfs:
from sklearn.metrics import accuracy_score
# df = df.iloc[0:, 1:4]
# df1= df1.iloc[0:,1:4]
#
# # print(df)
# from scipy import stats
# # import numpy as np
# #
# # np.random.seed(12345678)
# temp = df.mean(axis=0)
# temp1 =df1.mean(axis=0)
# print(temp)
# print(temp1)
# print(temp['x'])

# for i in range(0, len(df)):
#     x1 = x1 + df.iloc[:,]

# print("is it printing")
# print(temp1)
# print(temp)
# temps1 = [temp['x'], temp['y'], temp['z']]
# temps2 = [temp1['x'], temp1['y'], temp1['z']]
# print(temp1,temps2)
# res = stats.f_oneway(temps1, temps2)
#
# print(res)


""" # print("*****************KS Test*****************")
    # value, p_value = stats.ks_2samp(temps1, temps2)
    # print(p_value)

    # print("Anderson ")
    # res = stats.anderson_ksamp([temps1, temps2])
    # print(res)


    #Correlation between columns of two dataframes.
    # res = X.corrwith(X1, axis = 0, method='spearman')
    # print("Correlation with")
    # print(res)

    #Cosine Similarity
    from sklearn.metrics.pairwise import euclidean_distances

    from math import*


    def square_rooted(x):
        return round(sqrt(sum([a * a for a in x])), 3)

    def cosine_similarity(x, y):
        numerator = sum(a * b for a, b in zip(x, y))
        denominator = square_rooted(x) * square_rooted(y)
        return round(numerator / float(denominator), 3)

    #print("****************Cosine Similarity ****************")
    res = cosine_similarity(temps1,temps2)
    # print(res)

    from math import *
    from decimal import Decimal


    def manhattan_distance(x, y):
        return sum(abs(a - b) for a, b in zip(x, y))

    a = manhattan_distance(temps1, temps2)
    #print("************************Manhattan Distance******************")
    # print(a)
    #
    # print("********Euclidean Distance************")
    from scipy.spatial import distance

    res1 = distance.euclidean(temps1, temps2)
    # print(res1)
    # print("*******************************************")"""