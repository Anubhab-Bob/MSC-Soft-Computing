from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
x = [[1],
     [0],
     [1],
     [1]]
y = [0, 1]
clf = MLPClassifier(solver='lbfgs', activation='logistic',

                    alpha=1e-5,
                     hidden_layer_sizes=(4,3,3,2),
                    random_state=1,
                    learning_rate_init=0.9,
                    max_iter=200)
#Train the network
print(clf.fit(x, y))


#Plot predicted values
y_predict=clf.predict(x)
print(y_predict)

plt.plot(x,y_predict)
plt.legend(['original','Target'])
plt.show()


print(clf.predict([[2., 2.], [-1., -2.]]))
print(clf.predict_proba([[2., 2.], [1., 2.]]))