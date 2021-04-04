# Realization of a three inputs bipolar NOR gate using perceptron

"""Importing python library."""
import neurolab as nl
import numpy as np
import matplotlib.pyplot as plt

"""Create a variable named data that is a list that contains the eight possible inputs to an NOR gate."""

data = [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]]
labels = [[1], [-1], [-1], [-1], [-1], [-1], [-1], [-1]]

# Syntax for 3-D projection
fig = plt.figure()
ax = plt.axes(projection="3d")
for inp,target in zip(data,labels):
        plt.plot(inp[0],inp[1],inp[2],'ro' if (target == 1.0) else 'bo')
plt.legend(['1','-1'])
plt.title("3 input NOR gate")
plt.show()

net = nl.net.newp([[-1, 1],[-1, 1],[-1, 1]], 1)

error = net.train(data, labels, goal=0.1, epochs=1000, show=100, lr=0.01)

import pylab as pl
pl.plot(error)
pl.xlabel('Epoch number')
pl.ylabel('Train error')
pl.grid()
pl.show()