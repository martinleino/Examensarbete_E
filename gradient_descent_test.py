#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 13:29:13 2023

@author: Martin
"""

# import pandas as pd
import numpy as np
import sklearn.linear_model as skl_lm
import matplotlib.pyplot as plt

# url = 'https://uu-sml.github.io/course-sml-public/data/auto.csv'
# auto = pd.read_csv(url, na_values='?').dropna()

# data = np.random.normal(loc=[175, 70], scale=[5, 3], size=(100,2))
# data = np.random.normal(loc=[175, 70], scale=[5, 3], size=(100,2))
# print(y_train)
# X_train = data[:,0]
# X_train = X_train.reshape((100,1))
# y_train = data[:,1]

# Generate training data
X_train = np.random.normal(loc=100, scale=5, size=100)
y_train = 2*X_train + np.random.normal(loc=0, scale=5, size=100)
X_train = X_train - np.mean(X_train) # center x-values around mean
training_data = np.column_stack([X_train, y_train])

"""
Analytical solution
"""

# Create LinearRegression object, fit to training data, compute predictions
lr = skl_lm.LinearRegression()
lr.fit(X=X_train.reshape((100,1)),y=y_train)
y_hats = lr.predict(X=X_train.reshape((100,1)))

# Plot training data and fitted line
plt.close('all')
plt.scatter(X_train, y_train)
plt.plot(X_train, y_hats, 'r')
plt.show()

"""
Gradient descent solution
"""

theta_0 = np.array([float(lr.intercept_) + 5*np.random.rand(), float(lr.coef_[0]) + np.random.rand()]) # initial guess
# theta_0 = np.zeros((1,2))
# print(theta_0)
# theta_0_perturbed = theta_0 + np.array([0.02, 0.02]) # to satisfy first condition in while-loop when taking norm(theta_k - theta_{k-1})
# # print(theta_0_perturbed)
# thetas = np.row_stack([theta_0, theta_0_perturbed])
thetas = np.row_stack([theta_0, theta_0])
delta = 0.001 # step-size
n = len(training_data)
gradients = np.zeros((1,2)) # Just to check how the gradient changes
analytic_sol = np.array([float(lr.intercept_), float(lr.coef_[0])])

# for i in range(10000):
while np.linalg.norm(thetas[-1:]-analytic_sol) > 0.05:
    y_pred = thetas[-1,0] + thetas[-1,1]*X_train
    gradient = 2/n * np.array([sum(y_pred-y_train), sum((y_pred-y_train)*X_train)])
    gradients = np.row_stack([gradients, gradient])
    theta_hat = thetas[-1:] - delta * gradient
    thetas = np.row_stack([thetas, theta_hat])

print('The analytical solution is alpha =', round(lr.intercept_, 4), 'and beta =', round(float(lr.coef_), 4))
print('The estimated parameter values with gradient descent are alpha =',round(thetas[-1,0], 4), 'and beta =',round(thetas[-1,1], 4),'.')

# while np.linalg.norm(thetas[-1:]-thetas[-2:]) > 0.02:
# for i in range(5):
#     partial_alpha = sum([thetas[-1,0] + thetas[-1,1]*X_train[i] - y_train[i] for i in range(n)])
#     print(partial_alpha)
#     partial_beta = sum([(thetas[-1,0] + thetas[-1,1]*X_train[i] - y_train[i])*X_train[i] for i in range(n)])
#     print(partial_beta)
#     gradient = 2/n * np.array([partial_alpha, partial_beta])
#     print(gradient)
#     theta_hat = thetas[-1:] - delta * gradient
#     print(theta_hat)
#     thetas = np.row_stack([thetas, theta_hat])
#     print(thetas)

plt.figure()
plt.scatter(thetas[1:,0], thetas[1:,1])
plt.plot(lr.intercept_, lr.coef_[0], 'ro')
plt.xlabel('alpha (intercept)')
plt.ylabel('beta (slope)')
plt.title('Gradient descent trajectory')
plt.show()

"""
Stochastic gradient descent solution
"""
n_b = 10 # mini-batch size
n_batches = int(n/n_b) # number of batches
E = 10 # number of epochs
thetas_SGD = np.row_stack([theta_0, theta_0])

while np.linalg.norm(thetas_SGD[-1:]-analytic_sol) > 0.05:
    # for _ in range(E):
        print(np.linalg.norm(thetas_SGD[-1:]-analytic_sol))
        np.random.shuffle(training_data)
        for j in range(n_batches):
            indices = np.arange((j-1)*n_b, j*n_b)
            gradient = 2/n_b * np.array([(y_pred-y_train)[indices].sum(), ((y_pred-y_train)*X_train)[indices].sum()])
            theta_hat = thetas[-1:] - delta * gradient
            thetas_SGD = np.row_stack([thetas_SGD, theta_hat])
 
print('The estimated parameter values with SGD are alpha =',round(thetas_SGD[-1,0], 4), 'and beta =',round(thetas_SGD[-1,1], 4),'.')        
plt.figure()
plt.scatter(thetas_SGD[1:,0], thetas_SGD[1:,1])
plt.plot(lr.intercept_, lr.coef_[0], 'ro')
plt.xlabel('alpha (intercept)')
plt.ylabel('beta (slope)')
plt.title('SGD trajectory')
plt.show()
    


