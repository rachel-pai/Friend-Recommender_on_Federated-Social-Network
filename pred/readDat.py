# coding:utf-8
# Created by chen on 28/05/2018
# email: q.chen@student.utwente.nl

import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

dire_temp  = '../pred/temp/train'
score_matrix,flatten_matrix = [],[]
for filename in os.listdir(dire_temp):
    if filename.startswith("score_matirx_"):
        with open(os.path.join(dire_temp, filename), 'rb') as score_file:
            score_file.seek(0)
            score_matrix = pickle.load(score_file)
    elif filename.startswith("y_labels_"):
        with open(os.path.join(dire_temp, filename), 'rb') as flatten_file:
            flatten_file.seek(0)
            flatten_matrix = pickle.load(flatten_file)

# print(len(score_matrix))
# print(np.array(flatten_matrix).shape)
scoreDF = pd.DataFrame(score_matrix,columns=['RPR','simRank','katz_score','aa','social','CT','CST','CN','SC','HP','LHN','PD','RA'])
scoreDF.drop(columns=['aa', 'CN','SC','HP','LHN','PD','RA'])
flatten_matrix = np.array([flatten_matrix])
labelDF = pd.DataFrame(flatten_matrix.T,columns=['label'])
labelDF.loc[labelDF.label != -1, 'label'] = 1
labelDF.loc[labelDF.label == -1, 'label'] = 0
print(labelDF.loc[labelDF['label'] == -1])
# scoreDF.to_csv('score2.csv',index=False)
combined = pd.concat([labelDF, scoreDF], axis=1, sort=False)

combined.to_csv('pandas_train.csv', index=None)

from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint

#
# score_matrix = np.array(score_matrix).astype(float)
# print(score_matrix.shape)
# pca = PCA(n_components=2)
#
# comps = pca.fit_transform(score_matrix)
# from sklearn import svm
#
# X = comps
# y = np.matrix(flatten_matrix).transpose()
# from sklearn.model_selection import cross_val_score
#
# clf = svm.SVC(kernel='linear', C=1)
# scores = cross_val_score(clf, X, y, cv=10)
# # clf = svm.SVC()
# # clf.fit(X, y)
# with open('../pred/model/svm', 'wb') as fp:
#     pickle.dump(clf, fp)
# #clf2 = pickle.loads(s)
# # print(clf.predict([[1,2]]))
# print(scores)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))