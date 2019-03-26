'''
Exercise 1: Linear regression with multiple variables

Instructions
------------
This file contains code that helps you get started on the linear regression exercise. You will need to complete the following functions in this exercise:
   warmUpExercise.py
   plotData.py
   gradientDescent.py
   computeCost.py
   gradientDescentMulti.py
   computeCostMulti.py
   featureNormalize.py
   normalEqn.py

For this part of the exercise, you will need to change some parts of the code below for various experiments (e.g., changing learning rates).
'''

# Initialization
import numpy as np
np.set_printoptions(suppress=True)


'''
================ Part 1: Feature Normalization ================
import os
os.chdir("Y:\Online Learning\Machine Learning - Coursera\Programming Assignments\Python Replicates\machine-learning-ex1")
'''

print ("Loading data")
# Load Data
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:,:2]
Y = np.reshape(data[:,2],(len(data[:,2]), 1))
m = len(Y) 


# Print out some data points
print ("First 10 examples from the dataset: X: ")
print (X[:10,:])

print ("First 10 examples from the dataset: Y: ")
print (Y[:10,:])

input("Program paused. Press enter to continue.")

# Scale features and set them to zero mean
print ("Normalizing Features ...")

from featureNormalize import featureNormalize
X,mu,sigma = featureNormalize(X)

# Add intercept term to X
X = np.append(np.ones((len(X),1)),X, axis=1)




# ================ Part 2: Gradient Descent ================
print ("Running gradient descent ...")
'''
====================== YOUR CODE HERE ======================
Instructions: We have provided you with the following starter
              code that runs gradient descent with a particular
              learning rate (alpha). 
              Your task is to first make sure that your functions - 
              computeCost and gradientDescent already work with 
              this starter code and support multiple variables.
              After that, try running gradient descent with 
              different values of alpha and see which one gives
              you the best result.
              Finally, you should complete the code at the end
              to predict the price of a 1650 sq-ft, 3 br house.
Hint: At prediction, make sure you do the same feature normalization.
'''

# Choose some alpha value
alpha = 1.2
num_iters = 100

# Init Theta and Run Gradient Descent 
theta = np.zeros((3,1))

from gradientDescentMulti import gradientDescentMulti

theta, J_history = gradientDescentMulti(X, Y, theta, alpha, num_iters)

# Plot the convergence graph
import matplotlib.pyplot as plt
plt.plot(np.arange(0,num_iters,1), J_history, color = 'green')
plt.xlabel("Number of iterations")
plt.ylabel("Cost J")
plt.xticks(np.arange(0,num_iters,5))
plt.show(block=False)
plt.pause(5)
plt.close()

# Display gradient descent's result
print ("Theta computed from gradient descent: ")
print (theta)


'''
Estimate the price of a 1650 sq-ft, 3 br house
====================== YOUR CODE HERE ======================
Recall that the first column of X is all-ones. Thus, it does
not need to be normalized.
'''

price = ([1650, 3] - mu) / sigma
price = np.append(1,price)
price = np.matmul(price, theta)



# ============================================================
print ("Predicted price of a 1650 sq-ft, 3 br house (using gradient descent) is \n $%0.02f" % price)


input("Program paused. Press enter to continue.")




# ================ Part 3: Normal Equations ================

print ("Solving with normal equations")
'''
% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form 
%               solution for linear regression using the normal
%               equations. You should complete the code in 
%               normalEqn.m
%
%               After doing so, you should complete this code 
%               to predict the price of a 1650 sq-ft, 3 br house.
%
'''

# Load Data
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:,:2]
Y = np.reshape(data[:,2],(len(data[:,2]), 1))
m = len(Y) 

# Add intercept term to X
X = np.append(np.ones((len(X),1)),X, axis=1)

# Calculate the parameters from the normal equation
from normalEqn import normalEqn
theta = normalEqn(X, Y)

# Display normal equation's result
print ("Theta computed from the normal equations:")
print (theta)

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
price = [1, 1650, 3]
price = np.matmul(price, theta)

# ===========================================================
print ("Predicted price of a 1650 sq-ft, 3 br house (using gradient descent) is \n $%0.02f" % price)


