# coding:utf-8
# Created by chen on 28/05/2018
# email: q.chen@student.utwente.nl

import os
import pickle
import numpy as np
from sklearn.decomposition import PCA

dire_temp  = '../pred/temp/'
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


score_matrix = np.array(score_matrix).astype(float)
print(score_matrix.shape)
pca = PCA(n_components=2)

comps = pca.fit_transform(score_matrix)
from sklearn import svm

X = comps
y = np.matrix(flatten_matrix).transpose()
clf = svm.SVC()
clf.fit(X, y)
with open('../pred/model/svm', 'wb') as fp:
    pickle.dump(clf, fp)
#clf2 = pickle.loads(s)
# print(clf.predict([[1,2]]))