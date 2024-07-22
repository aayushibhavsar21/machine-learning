#_________ applying gradient descent algorithm to find equation for given data _________

def gradient_descent1(x,y):
    m_curr = b_curr = 0   # starting points
    iteration = 1700      # take random number of iterations (strps) to reach on minimum MSE
    learning_rate = 0.08  # take random minimum value as learning rate (eg = 0.0001) , for learning rate cost should be decrese at every step 
                          # we can change this value to max random for which cost decrese at every step
                          # for 0.08 cost decrese at each step but for 0.09 cost started to increse. so we will use 0.08
    n = len(x)   

    for i in range(iteration):

        y_predicted = m_curr * x + b_curr                          # y = mx + c

        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])    # MSE

        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)

        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd

        print ("m {}, b {}, cost {} iteration {}".format(m_curr,b_curr,cost, i))

import numpy as np
x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradient_descent1(x,y)




# _________finding relation between 2 variable of CSV file_________

def gradient_descent2(x,y):
    m_curr = b_curr = 0     # starting points
    iteration = 1000000     # take random number of iterations (strps) to reach on minimum MSE
    learning_rate = 0.0002  # take random minimum value as learning rate (eg = 0.0001) , for learning rate cost should be decrese at every step 
                            # we can change this value to max random for which cost decrese at every step
                            # for 0.08 cost decrese at each step but for 0.09 cost increse so we will use 0.08
    n = len(x)   

    for i in range(iteration):

        y_predicted = m_curr * x + b_curr                          # y = mx + c

        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])    # MSE

        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)

        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd

        print ("m {}, b {}, cost {} iteration {}".format(m_curr,b_curr,cost, i))

import pandas as pd
df = pd.read_csv("A:/machine learning/gradient descent/test_scores.csv")
print(df.math,"\n",df.cs)

gradient_descent2(df.math,df.cs)

