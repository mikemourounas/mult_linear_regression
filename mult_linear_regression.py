'''
Demonstration of multiple linear regression in python

Author: Michael Mourounas

Dataset: Boston house prices dataset
https://archive.ics.uci.edu/ml/machine-learning-databases/housing/

Date: 16/7/2019

Code written in Sublime Text
'''

# Import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
from sklearn.datasets import load_boston

boston = load_boston()

X = pd.DataFrame(data=boston.data, columns=boston.feature_names)
X = pd.get_dummies(X, columns=['CHAS'])  # OneHotEncode the dummy var
y = pd.Series(boston.target, name='MEDV')

# Build and train test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Fit regression model
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Initial predictions
y_pred = pd.Series(data=lin_reg.predict(X_test), name='MEDV_PRED')

# Define backward elimination function
import statsmodels.api as sm


def backward_elimination(y, x, SL):
    '''
    for a given y vector, matrix of features x and significance threshold,
    perform step-wise regression starting with all possible features until
    the model contains only those features within the given SL
    '''
    num_vars = len(x[0])
    temp = np.zeros(x.shape).astype(int)
    for i in range(0, num_vars):
        lin_reg_OLS = sm.OLS(y, x).fit()
        print(lin_reg_OLS.summary())
        max_var = max(lin_reg_OLS.pvalues).astype(float)
        adjR_before = lin_reg_OLS.rsquared_adj.astype(float)
        if max_var > SL:
            for j in range(0, num_vars - i):
                if lin_reg_OLS.pvalues[j].astype(float) == max_var:
                    temp[:, j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_reg = sm.OLS(y, x).fit()
                    adjR_after = tmp_reg.rsquared_adj.astype(float)
                    if adjR_before >= adjR_after:
                        x_rollback = np.hstack((x, temp[:, [0, j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print(lin_reg_OLS.summary())
                        return x_rollback
                    else:
                        continue
    print(lin_reg_OLS.summary())
    return x

# Perform backward elimination
X_vals = X.iloc[:, :].values
X_vals = np.append(arr=np.ones(shape=(506, 1)).astype(int),
                   values=X_vals, axis=1)
y_vals = y.values

X_modeled = backward_elimination(y_vals, X_vals, SL=0.05)

# TO DO

# Validate stepwise function manually
# Fit sklearn model to optimized matrix of features
# Confirm linear assumptions
# Visualize the data
