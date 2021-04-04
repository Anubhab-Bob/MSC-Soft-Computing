# -*- coding: utf-8 -*-
"""
@author: Anubhab
"""
# import necessary libraries
from sklearn.linear_model import Perceptron
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate the accuracy of the model
def get_accuracy(test_inp, test_out, true_inp, true_out):
    index = -1
    for i in range(8):
        if((true_inp[i] == test_inp).all()):
            index = i
            break
        #index = np.where((true_inp[i] == test_inp).all())
    if test_out == true_out[index]:
        return 1
    else:
        return 0

# NOR gate inputs and outputs
data = [
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, 1, 1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, -1],
        [1, 1, 1]
        ]
labels = [1, -1, -1, -1, -1, -1, -1, -1]

# 3-D projection of 3-input NOR gate
print("Displaying 3-i/p NOR gate logic values ...")
fig = plt.figure()
ax = plt.axes(projection="3d")
for inp,target in zip(data,labels):
        plt.plot(inp[0],inp[1],inp[2],'ro' if (target == 1.0) else 'bo')
plt.legend(['1','-1'])
plt.title("3 input NOR gate")
plt.show()

# Perceptron to learn NOR.
classifier = Perceptron(max_iter = 40)
classifier.fit(data, labels)

#Training Complete
print("Training complete - ",str(classifier.score(data,labels)*100),"%")

# Testing
test_n = int(input("Enter the number of test samples : "))
test = np.random.randint(2, size=(test_n, 3))
for i in range(test_n):
    test[i] = [-1 if i==0 else i for i in test[i]]

prediction = classifier.predict(test)
accuracy = 0
print("\nTest Input\tPrediction")
print("--------------------------------------")
for i in range(test_n):
    print(str(test[i]),"\t",prediction[i])
for i in range(test_n):
    accuracy += get_accuracy(test[i],prediction[i], data, labels)

accuracy = (accuracy/test_n) * 100
print("\nAccuracy of model : %3.2f"%accuracy+"%")

# Prediction plot
fig = plt.figure()
ax = plt.axes(projection="3d")
for inp,target in zip(test,prediction):
        plt.plot(inp[0],inp[1],inp[2],'mo' if (target == 1.0) else 'go')
plt.legend(['1','-1'])
plt.title("3 input NOR gate Perceptron Predictions")
plt.show()