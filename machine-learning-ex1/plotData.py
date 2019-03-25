
'''
Instructions: 
Plot the training data into a figure. 
Set the axes labels using the "xlabel" and "ylabel" commands. 
Assume the population and revenue data have been passed in as the x and y arguments of this function.
'''

import matplotlib.pyplot as plt
def plotData(X,Y):
    plt.scatter(X,Y,c='red',marker='X')
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.show(block=False)
    plt.pause(5)
    plt.close()

