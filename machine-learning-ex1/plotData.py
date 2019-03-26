
'''
PLOTDATA Plots the data points x and y into a new figure 

PLOTDATA(x,y) plots the data points and gives the figure axes labels of population and profit.

'''

import matplotlib.pyplot as plt
def plotData(X,Y):
    # ====================== YOUR CODE HERE ======================
    # Instructions: Plot the training data into a figure.
    #               Set the axes labels using the "xlabel" and "ylabel" . 
    #               Assume the population and revenue data have been passed in
    #               as the x and y arguments of this function.
    #

    plt.scatter(X,Y,c='red',marker='X')
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.show(block=False)
    plt.pause(5)
    plt.close()


    # ============================================================