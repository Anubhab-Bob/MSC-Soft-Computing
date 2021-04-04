#!/usr/bin/env python
# coding: utf-8
"""
@author: Anubhab
"""
# Implementation of 3-2-1 MADALINE MR-1 Learning

# import necessary libraries
import numpy as np

# function for madaline learning
def MAD(i, w, x, t, h):

    z_in = x[i].dot(w) #Matrix Multiplication Using Dot
    z_out = np.where(z_in>= 0, 1, -1) #Check all z_in values --> if z_in[n]>0 then 1 else -1
    y_in = h + h * z_out[0] + h * z_out[1]
    if y_in>= 0:
        y_out = 1
    else:
        y_out = -1
    if y_out != t[i]:
        if t[i] == -1:
            z_in[z_in < 0] = -1
            w = w + h * (-1 - z_in) * x[i].reshape(-1, 1)
        if t[i] == 1:
            k = min(z_in, key=abs)
            z_in[z_in != k] = 1  # Minimum Value
            w = w + h * (1 - z_in) * x[i].reshape(-1, 1)
            w = np.round(w, 2)
    else:
        print("Found match at", i+1, "of epoch", epoch+1)
        y[i] = y_out
    return w, y

# input data
x=np.array([[1,1,1,1],
            [1,1,1,-1],
            [1,1,-1,1],
            [1,1,-1,-1],
            [1,-1, 1, 1],
            [1,-1, 1, -1],
            [1,-1, -1, 1],
            [1,-1, -1, -1]])
print("Given Input Set is:\n",x)
#Output Set
t=np.array([1, -1, -1, 1, -1, 1, 1, 1])
#Set for learning predictions
y=np.array([0, 0, 0, 0, 0, 0, 0, 0])
#Weights of the operation #Random Value
weight=np.array([[0.2, 0.3],
                 [0.3,0.2],
                 [0.2,0.1],
                 [0.1,0.1]])
h = float(input("Enter Learning Rate: "))
check = False
for epoch in range(0, 10):
    print("#EPOCH => ", epoch+1)
    if np.array_equal(y, t):   #Check if array(y) = array(t)
        check = True
        break
    else:
        for i in range(0, 8):
            weight, y = MAD(i, weight, x, t, h)

#Check Output
print("Actual Output values:\n", t)
print("Predicted Output values:\n", y)
if check:
    print("The Output is reached at EPOCH ", epoch + 1)
else:
    print("EPOCH Limit Reached!! Please Retrain")