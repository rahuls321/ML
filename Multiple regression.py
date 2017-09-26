# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 18:30:24 2017

@author: hp
"""


import matplotlib.pyplot as plt
from sklearn import datasets , linear_model , metrics
from sklearn.datasets import load_boston
boston =load_boston(return_X_y= False)

X = boston.data
y = boston.target

from sklearn.model_selection import train_test_split

X_train, X_test , y_train, y_test = train_test_split(X, y, test_size = 0.4 ,random_state = 1)

reg = linear_model.LinearRegression()

reg.fit(X_train , y_train)

print('Coefficients: \n', reg.coef_)

print ("Variance Score: {} \n".format(reg.score(X_test, y_test)))

# plot for residual error

plt.style.use('fivethirtyeight')
plt.scatter(reg.predict(X_train) , reg.predict(X_train) - y_train , color = 'green', s = 10, label ='Train data' )

plt.scatter(reg.predict(X_test) , reg.predict(X_test) - y_test , color = 'green', s = 10, label ='Test data' )

plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2 )
plt.legend(loc = 'upper right')

plt.title('Residual errors')
plt.show()