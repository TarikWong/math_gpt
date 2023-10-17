# -*- coding: utf-8 -*-
# @Time : 2023/9/21 10:12 上午
# @Author : tuo.wang
# @Version :
# @Function :
from __future__ import print_function
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from pprint import pprint
import logging
from time import time
import numpy as np
import os
from sklearn.cluster import DBSCAN
import pandas as pd
import jieba

file_path = '/Users/tuo/Downloads/none_mapping.json'
output_file_path = "{}.csv".format(file_path.split('.')[0])

df_dict = {}
org_data = []
data = []
with open(file_path, 'r') as f:
    for line in f.readlines():
        if line.strip() != '{' and line.strip() != '}':
            # print(line)
            org_data.append(line.strip())
            data.append(line.strip().replace('": [],', '').replace('": []', '').replace('"', ''))
df_dict['org_data'] = org_data
df_dict['data'] = data
input_df = pd.DataFrame(df_dict, columns=['org_data', 'data'])
# print(input_df)

# 去掉文本为空的数据
null = input_df['data'].isnull()
no_null = ~null
input_nonull_df = input_df[no_null]
print("The length of input_nonull_df is: ", len(input_nonull_df))
print("##############################################################################")

# 去掉文本重复的
input_nonull_unique_df = input_nonull_df.drop_duplicates('data')
print("The length of input_nonull_unique_df is: ", len(input_nonull_unique_df))
print("##############################################################################")

# 分词
jieba.enable_parallel()
input_nonull_unique_df['data_cut'] = input_nonull_unique_df['data'].apply(lambda i: jieba.lcut(i))
input_nonull_unique_df['data_cut'] = [' '.join(i) for i in input_nonull_unique_df['data_cut']]
print(input_nonull_unique_df['data_cut'][:10])
print("##############################################################################")

# 提取稀疏文本特征
print("使用稀疏向量（Sparse Vectorizer）从训练集中抽取特征")
t0 = time()

vectorizer = TfidfVectorizer(max_df=0.5,
                             min_df=20,
                             max_features=40000,
                             ngram_range=(1, 2),
                             use_idf=True)

X = vectorizer.fit_transform(input_nonull_unique_df['data_cut'])

print("完成所耗费时间： %fs" % (time() - t0))
print("样本数量: %d, 特征数量: %d" % X.shape)
print('特征抽取完成！')
print("##############################################################################")

# 降维，使其适用于DBSCAN算法
print("用LSA进行维度规约（降维）...")
t0 = time()

# Vectorizer的结果被归一化，这使得KMeans表现为球形k均值（Spherical K-means）以获得更好的结果。
# 由于LSA / SVD结果并未标准化，我们必须重做标准化。

svd = TruncatedSVD(25)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X = lsa.fit_transform(X)

print("完成所耗费时间： %fs" % (time() - t0))

explained_variance = svd.explained_variance_ratio_.sum()
print("SVD解释方差的step: {}%".format(int(explained_variance * 100)))

print('PCA文本特征抽取完成！')
print("##############################################################################")

# 进行实质性的DBScan聚类
db = DBSCAN(eps=0.3, min_samples=50).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

# print(db.core_sample_indices_)

labels = db.labels_
# print(labels)

clusterTitles = db.labels_
dbscandf = input_df
dbscandf['cluster'] = clusterTitles
dbscandf.sort_values(by="cluster", axis=0, ascending=True, inplace=True)
dbscandf.to_csv(output_file_path, index=False)

# 看看簇群序号为0的文章的标题有哪些，通过这个能看出聚类的实际效果如何
print('簇群tag为0的title名称')
print(dbscandf[dbscandf['cluster'] == 0]['data'].head(20))  # 簇群tag为0的title名称
print("##############################################################################")

# 看看簇群序号为20的文章的标题有哪些，通过这个能看出聚类的实际效果如何
print('簇群tag为20的title名称')
print(dbscandf[dbscandf['cluster'] == 20]['data'].head(20))  # 簇群tag为20的title名称

# 聚类数及噪点计算
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('聚类数：', n_clusters_)
print('噪点数：', n_noise_)
print("##############################################################################")

# 对结果可视化
import matplotlib.pyplot as plt

# 黑色点是噪点，不参与聚类
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('cluster num: %d' % n_clusters_)
plt.show()
print('done.')
