import math

import pandas as pd
import matplotlib.pyplot as plt


import scipy.stats as stats

df =pd.read_csv('C:/Users/Siri/PycharmProjects/Dataset_conversion/Pascal_VOC/classes_2007/boat/metadata_features_tsne.tsv',delimiter='\t',encoding='utf-8')
df['label'] = 0

df1 =pd.read_csv("C:/Users/Siri/PycharmProjects/Dataset_conversion/Pascal_VOC/classes_2007/diningtable/metadata_features_tsne.tsv",delimiter='\t',encoding='utf-8')
df1['label'] = 1

df2 =pd.read_csv("C:/Users/Siri/PycharmProjects/Dataset_conversion/Pascal_VOC/classes_2007/horse/metadata_features_tsne.tsv",delimiter='\t',encoding='utf-8')
df2['label'] = 2

df3 =pd.read_csv("C:/Users/Siri/PycharmProjects/Dataset_conversion/Pascal_VOC/classes_2007/person/metadata_features_tsne.tsv",delimiter='\t',encoding='utf-8')
df3['label'] = 3
#
# df4 =pd.read_csv("C:/Users/Siri/PycharmProjects/Dataset_conversion/Pascal_VOC/classes_2007/chair/metadata_features_tsne.tsv",delimiter='\t',encoding='utf-8')
# df4['label'] = 4
#
# df5 =pd.read_csv("C:/Users/Siri/PycharmProjects/Dataset_conversion/Pascal_VOC/classes_2007/cow/metadata_features_tsne.tsv",delimiter='\t',encoding='utf-8')
# df5['label'] = 5
#
# df6 =pd.read_csv("C:/Users/Siri/PycharmProjects/Dataset_conversion/Pascal_VOC/classes_2007/sheep/metadata_features_tsne.tsv",delimiter='\t',encoding='utf-8')
# df6['label'] = 6
#
# df7 =pd.read_csv("C:/Users/Siri/PycharmProjects/Dataset_conversion/Pascal_VOC/classes_2007/sofa/metadata_features_tsne.tsv",delimiter='\t',encoding='utf-8')
# df7['label'] = 7

t = df.mean(axis=0)
t1 =df1.mean(axis=0)
t2 = df2.mean(axis=0)
t3 =df3.mean(axis=0)
# t4 =df4.mean(axis=0)
# t5 =df5.mean(axis=0)
# t6 =df6.mean(axis=0)
# t7 =df7.mean(axis=0)


ts = [t['x'], t['y'], t['z']]
ts1 = [t1['x'], t1['y'], t1['z']]
ts2 = [t2['x'], t2['y'], t2['z']]
ts3 = [t3['x'], t3['y'], t3['z']]
# ts4 = [t4['x'], t4['y'], t4['z']]
# ts5 = [t5['x'], t5['y'], t5['z']]
# ts6 = [t6['x'], t6['y'], t6['z']]
# ts7 = [t7['x'], t7['y'], t7['z']]

#Plotting TSNE Datasets:
plt.figure(figsize=(116,150))
plt.title("TSNE Visualization")
plt.scatter(df.iloc[:,1],df.iloc[:,2],  c='b', label ='boat')
plt.scatter(df1.iloc[:,1],df1.iloc[:,2], c='r',label='diningtable')
plt.scatter(df2.iloc[:,1],df2.iloc[:,2],c='g',label ='horse')
plt.scatter(df3.iloc[:,1],df3.iloc[:,2], c='y', label='person')
# plt.scatter(df4.iloc[:,1],df4.iloc[:,2],c='m',label='chair')
# plt.scatter(df5.iloc[:,1],df5.iloc[:,2], c='c', label ='cow')
# plt.scatter(df6.iloc[:,1],df6.iloc[:,2],c=(0.69411766529083252, 0.3490196168422699, 0.15686275064945221, 1.0),  label='sheep')
# plt.scatter(df7.iloc[:,1],df7.iloc[:,2],c=(0.65098041296005249, 0.80784314870834351, 0.89019608497619629, 1.0),  label='sofa')
plt.legend(loc='upper left')
#plt.savefig('C:/Users/Siri/Desktop/Final Thesis Results/Graphs/Clustering/Group-1/group_1.jpg')
plt.show()


plt.figure(figsize=(116,150))
plt.title("Mean Visualization")
plt.scatter(t[0],t[1],  c='b', label ='boat')
plt.scatter(t1[0],t1[1], c='r',label='diningtable')
plt.scatter(t2[0],t2[1],c='g',label ='horse')
plt.scatter(t3[0],t3[1], c='y', label='person')
plt.show()






