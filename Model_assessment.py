#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: hbasriaykol

#K-fold Cross Validation
"""
Allows all datas to be samples.
Example, For cv =4 ;
(3train/1test), (1test/3train),(1train,1test,2train) ..
Results will be different, you can look over average or standard deviations

"""
#svc ; support vector machine
#x_train, y_train ; To split data : x_train, y_train, x_test, y_test ..
from sklearn.model_selection import cross_val_score
cvs = cross_val_score(estimator = svc, x=x_train, y=y_train, cv = 4)
cvs.mean() , cvs.std()

#GridSearchCV
from sklearn.model_selection import GridSearchCV
p = [{'C' : [1,2,3,4,5], 'kernel': ['linear','rbf']}, 
    {'C' : [1,10,100,1000], 'kernel' : ['rbf']}] #Two case


gs = GridSearchCV(estimator = svc, param_grid = p , scoring = 'accuracy', cv = 10, n_jobs = -1)
grid_search = gs.fit(x_train, y_train)
thebestresult = grid_search.best_score_
thebestparameters = grid_search.best_params_

#estimator : which model we will optimize
#param_grid : paremeters
#scoring : score according to what
#cv : how many folds
#n_jobs : work at the same time


#Another Evaluation Types

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import LeaveOneOut