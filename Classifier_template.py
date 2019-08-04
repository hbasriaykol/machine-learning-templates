#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: hbasriaykol

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
datas = pd.read_excel('Iris.xls')

x = datas.iloc[:,:4]
y = datas.iloc[:,-1:]

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.33, random_state=0)


#logistic Regresyon Classifier
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0)
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

cm=confusion_matrix(y_test, y_pred)
print('\n' + 'Logistic  Regression' + '\n',cm)


#K-NN Classifier
from sklearn.neighbors import KNeighborsClassifier
knn  = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
knn2 = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn3 = KNeighborsClassifier(n_neighbors=5, metric='chebyshev')
knn4 = KNeighborsClassifier(n_neighbors=5, metric='manhattan')

knn.fit(x_train, y_train)
knn2.fit(x_train, y_train)
knn3.fit(x_train, y_train)
knn4.fit(x_train, y_train)

y_pred2 = knn.predict(x_test)
y_pred3 = knn2.predict(x_test)
y_pred4 = knn3.predict(x_test)
y_pred5 = knn4.predict(x_test)

cm2=confusion_matrix(y_test, y_pred2)
print('\n' + 'KNN-minkowski' + '\n', cm2)
cm3=confusion_matrix(y_test, y_pred3)
print('\n' + 'KNN-euclidean' + '\n', cm3)
cm4=confusion_matrix(y_test, y_pred4)
print('\n' + 'KNN-chebyshev' + '\n', cm4)
cm5=confusion_matrix(y_test, y_pred5)
print('\n' + 'KNN-manhattan' + '\n', cm5)


#Support Vector Machine Classifier
from sklearn.svm import SVC
svc=SVC(kernel='rbf')
svc2=SVC(kernel='poly')
svc3=SVC(kernel='sigmoid')
svc4=SVC(kernel='linear')

svc.fit(x_train, y_train)
svc2.fit(x_train, y_train)
svc3.fit(x_train, y_train)
svc4.fit(x_train, y_train)

y_pred6 = svc.predict(x_test)
y_pred7 = svc2.predict(x_test)
y_pred8 = svc3.predict(x_test)
y_pred9 = svc4.predict(x_test)

cm6=confusion_matrix(y_test, y_pred6)
print('\n' + 'SVC-rbf' + '\n', cm6)
cm7=confusion_matrix(y_test, y_pred7)
print('\n' + 'SVC-poly' + '\n', cm7)
cm8=confusion_matrix(y_test, y_pred8)
print('\n' + 'SVC-sigmoid ' + '\n', cm8)
cm9=confusion_matrix(y_test, y_pred9)
print('\n' + 'SVC-linear' + '\n', cm9)


#Naive Bayes Classifier
'''GaussianNB'''
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(x_train, y_train)

y_pred10 = gnb.predict(x_test)

cm10=confusion_matrix(y_test, y_pred10)
print('\n' + 'Naive Bayes-GaussianNB' + '\n', cm10)


'''MultinomialNB'''
from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB()
mnb.fit(x_train, y_train)

y_pred11 = mnb.predict(x_test)

cm11=confusion_matrix(y_test, y_pred11)
print('\n' + 'Naive Bayes-MultinomialNB' + '\n', cm11)


'''ComplementNB'''
from sklearn.naive_bayes import ComplementNB
cnb=ComplementNB()
cnb.fit(x_train, y_train)

y_pred12 = cnb.predict(x_test)

cm12=confusion_matrix(y_test, y_pred12)
print('\n' + 'Naive Bayes-ComplementNB' + '\n', cm12)


'''BernoulliNB'''
from sklearn.naive_bayes import BernoulliNB
bnb=BernoulliNB()
bnb.fit(x_train, y_train)

y_pred13 = bnb.predict(x_test)

cm13=confusion_matrix(y_test, y_pred13)
print('\n' + 'Naive Bayes-BernoulliNB' + '\n', cm13)


#Decision Tree Classifier
'''entropy'''
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion='entropy')
dtc.fit(x_train, y_train)

y_pred14 = dtc.predict(x_test)

cm14=confusion_matrix(y_test, y_pred14)
print('\n' + 'DecisionTree-Entropy' + '\n', cm14)


'''gini'''
dtc2=DecisionTreeClassifier(criterion = 'gini')
dtc2.fit(x_train, y_train)

y_pred15 = dtc2.predict(x_test)

cm15=confusion_matrix(y_test, y_pred15)
print('\n' + 'DecisionTree-Gini' + '\n', cm15)


#RandomForest Classifier
''' Entorpy'''
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=20, criterion='entropy')
rfc.fit(x_train, y_train)

y_pred16 = rfc.predict(x_test)

cm16=confusion_matrix(y_test, y_pred16)
print('\n' + 'RandomForest-Entropy' + '\n', cm16)


'''Gini'''
rfc2=RandomForestClassifier(n_estimators=20, criterion='gini')
rfc2.fit(x_train, y_train)

y_pred17 = rfc2.predict(x_test)

cm17=confusion_matrix(y_test, y_pred17)
print('\n' + 'RandomForest-Gini' + '\n', cm17)