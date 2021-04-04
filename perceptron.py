import numpy as np
class Perceptron:

    def __init__(self, learning_rate=0.01, epochs=100, threshold=1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None
        self.threshold = threshold


    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialise weights and bias
        self.weights = np.random.rand(n_features)
        self.bias = np.random.rand()

        y_ = np.array([1 / (1 + np.exp(-i)) for i in y])

        # Train the data
        for _ in range(self.epochs):
            for ido, o_i in enumerate(X):
                linear_output = np.dot(o_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                # Calculate updated values of weights and bias
                update = self.learning_rate * (y_[ido] - y_predicted)
                self.weights += update * o_i
                self.bias += update


    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted


    # Defining Activation Function
    def _unit_step_func(self, x):
        return np.array((np.where(x>self.threshold, 1),
                         (np.where(x<self.threshold, -1), 0), 0))