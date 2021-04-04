#!/usr/bin/env python
# coding: utf-8

# # Write a python script to train a 4-3-3-2 feed forward neural network using back propagation learning where the training pattern is {1,0,1,1} and output is (0,1}.
# The following code trains a multilayer feed-forward neural network using the back propagation algorithm . It iteratively learns a set of weights for prediction of the class label of tuples. A multilayer feed-forward neural network consists of an input layer, one or more hidden layers, and an output layer. For the given problem the layers and inputs are static and hard coded. The momentum is set to 0.3 which introduces some balancing in the update between the eigenvectors associated to lower and larger eigenvalues allowing the attenuation of oscillations in the gradient descent.

# In[10]:


# import necessary libraries
from neuralnetwork.FeedForward import FeedForward
from neuralnetwork.Sigmoid import Sigmoid
from neuralnetwork.Backpropagation import Backpropagation
import numpy as np
sigmoid = Sigmoid()


# In[11]:


# Build the model
networkLayer = [4,3,3,2]
feedForward = FeedForward(networkLayer, sigmoid)
lr = float(input("Enter the learning rate : "))
backpropagation = Backpropagation(feedForward,lr,0.3)


# In[12]:


# Train the model
trainingSet = [
    [1,0,1,1,0,1]
  ]
while True:
    backpropagation.initialise()
    result = backpropagation.train(trainingSet)
    if(result):
        break


# In[13]:


# Test the model
feedForward.activate([1,0,1,1])
outputs = np.array(feedForward.getOutputs())

label = np.array(trainingSet[0][4:])
#Display the true and generated outputs
print("Expected Output: ",label)
print("Actual Output: ", outputs)
error = [abs(i) for i in label-outputs]  # Calculate the final error
accuracy = np.mean([1-i for i in error]) * 100  # Calculate the accuracy %
print("Accuracy:\n%4.2f"%accuracy+"%")

