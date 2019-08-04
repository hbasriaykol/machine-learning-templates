#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: hbasriaykol

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import statsmodels.api as sm #To calculate p-value
from sklearn.metrics import r2_score

data = pd.read_csv('data.csv')

print("Correlation Between Variables\n" , data.corr())

#data[:,:1]   -> Overfitting
#data[:,1:2]  -> Dummy Variable
employee_data = data.iloc[:,2:5].values
sales = data.iloc[:,-1:].values

#For SVR
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
sc2 = StandardScaler()
employee_data_scale = sc1.fit_transform(employee_data)
sales_scale = sc2.fit_transform(sales)

###############################################################################

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(employee_data,sales)
print(r2_score(sales, lr.predict((employee_data))))

###############################################################################
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=4)
x_poly = pf.fit_transform(employee_data)
lr2=LinearRegression()
lr2.fit(x_poly,sales)
print(r2_score(sales, lr2.predict((x_poly))))
###############################################################################

from sklearn.svm import SVR
svrreg = SVR(kernel = 'rbf')
svrreg.fit(employee_data_scale,sales_scale)
print(r2_score(sales_scale, svrreg.predict((employee_data_scale))))
#############################################################

from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor(random_state=0)
dtr.fit(employee_data,sales)
print(r2_score(sales, dtr.predict((employee_data))))
#############################################################

from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(n_estimators = 10 , random_state=0)
rfr.fit(employee_data,sales)
print(r2_score(sales, rfr.predict((employee_data))))