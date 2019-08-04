#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: hbasriaykol

import pandas as pd
import numpy as np

datas = pd.read_csv("/your_data_address")
data = datas.iloc[:,:].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
data[:,1:6] = imputer.fit_transform(data[:,1:6])
#SimpleImputer fills only values containing numbers
#In my example only data[:,1:6] contains numbers

#Label Encoder 
"""
To convert categorical text data 
into model-understandable numerical data
Example ;
Male = [1]
Female = [0]

"""
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
encoder = ohe.fit_transform(data[:,0:1]).toarray()

#One Hot Encoder
"""
it takes a column which has categorical data, 
which has been label encoded, and then splits the column into multiple columns.
The numbers are replaced by 1s and 0s, depending on which column has what value.
Example ;
Turkey =[1,0,0] 
Germany = [0,1,0]
Spain = [0,0,1]

"""

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
encoder = ohe.fit_transform(data[:,0:1]).toarray()