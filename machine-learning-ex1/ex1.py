'''
Exercise 1: Linear Regression

Instructions

This file contains code that helps you get started on the linear exercise. You will need to complete the following functions in this exercise:

warmUpExercise.py
plotData.py
gradientDescent.py
computeCost.py
gradientDescentMulti.py
computeCostMulti.py
featureNormalize.py
normalEqn.py

For this exercise, you will not need to change any code in this file, or any other files other than those mentioned above. 

x refers to the population size in 10,000s 
y refers to the profit in $10,000s 
'''
import numpy as np

'''
==================== Part 1: Basic Function ====================
Complete warmUpExercise.py
'''

print ("Running Part 1: warmUpExercise ...")
print ("5x5 Identity Matrix:")

from warmUpExercise import warmUpExercise
warmUpExercise()
input("Program paused. Press enter to continue.")
'''
======================= Part 2: Plotting =======================
Complete plotData.py
'''

print ("Plotting Data ...")


data = np.loadtxt('ex1data1.txt', delimiter=',')
X = data[:,0]
Y = data[:,1]
m = len(data) 
from plotData import plotData
plotData(X, Y)
input("Program paused. Press enter to continue.")

'''
=================== Part 3: Cost and Gradient descent ===================
Complete computeCost.py
Complete gradientDescent.py
'''
X = np.reshape(data[:,0],(len(data[:,0]), 1))
Y = np.reshape(data[:,1],(len(data[:,1]), 1))
X = np.append(np.ones((len(X),1)),X, axis=1) # Add a column of ones to x
theta = np.zeros((2,1)) # initialize fitting parameters
# Some gradient descent settings
iterations = 1500
alpha = 0.01

print ("Testing the cost function ...")
# Compute and display initial cost
from computeCost import computeCost
J = computeCost(X,Y,theta)

print ("With theta = [0 ; 0]\n Cost computed = %0.2f" % J)
print ("Expected cost value (approx) 32.07")

# further testing of the cost function
J = computeCost(X, Y, np.array([[-1], [2]]))
print ("With theta = [-1 ; 2]\n Cost computed = %0.2f " % J)
print ("Expected cost value (approx) 54.24")

input("Program paused. Press enter to continue.")


print ("Running Gradient Descent ...")
# run gradient descent
from gradientDescent import gradientDescent
theta = gradientDescent(X, Y, theta, alpha, iterations)

# print theta to screen
print ("Theta found by gradient descent:\n %0.4f\n %0.4f" % (theta[0], theta[1]))
print ("Expected theta values (approx)\n -3.6303\n  1.1664")


# Plot the linear fit
import matplotlib.pyplot as plt
plt.scatter(X[:,1],Y,c='red',marker='x', label = 'Training data')
plt.plot(X[:,1], np.matmul(X, theta), c= 'blue', label = 'Linear regression')
plt.legend(loc = 'lower right')
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.xlim([4,24])
plt.xticks(np.arange(4,24,2))
plt.ylim([-5,25])
plt.show(block=False)
plt.pause(5)
plt.close()


# Predict values for population sizes of 35,000 and 70,000
predict1 = 10000 * np.matmul(np.array([1, 3.5]), theta)
print ("For population = 35,000, we predict a profit of %0.2f" % predict1)
predict2 = 10000 * np.matmul(np.array([1, 7]), theta)
print ("For population = 70,000, we predict a profit of %0.2f" % predict2)


input("Program paused. Press enter to continue.")



#============= Part 4: Visualizing J(theta_0, theta_1) =============



print ("Visualizing J(theta_0, theta_1) ...")
# Grid over which we will calculate J
theta0_vals = np.linspace(-10,10,100)
theta1_vals = np.linspace(-1,4,100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
from computeCost import computeCost
for i in range (len(theta0_vals)):
    for j in range (len(theta1_vals)):
        t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
        J_vals[i,j] = computeCost(X, Y, t)


# Surface Plot
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap = 'coolwarm',
                       linewidth=0, antialiased=False)
plt.xlabel("theta_0")
plt.ylabel("theta_1")
plt.xticks(np.arange(-10,10,5))
plt.show(block = False)
plt.pause(5)
plt.close()

# Contour plot
plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2,3,num=20))
plt.plot(theta[0], theta[1], marker = 'x', color = 'red')
plt.xlabel("theta_0")
plt.ylabel("theta_1")
plt.xticks(np.arange(-10,10,2))
plt.yticks(np.arange(-1,4,0.5))
plt.show(block = False)
plt.pause(5)
plt.close()
