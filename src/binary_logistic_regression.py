import numpy as np

class BinaryLogisticRegression:

    def __init__(self, dim):
        self.w = np.random.normal(size=(dim,))
        self.b = np.random.normal()

    def computeProb(self, X):
        z = np.dot(X, self.w) + self.b
        prob = self.sigmoid(z)
        return prob

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, X, threshold=0.5):
        probas = self.computeProb(X)
        preds = (probas > threshold).astype(np.int32)
        return preds

    def computeCost(self, X, y):
        prob = self.computeProb(X)
        J = -np.mean(y * np.log(prob) + (1. - y) * np.log(1. - prob))
        return J

    def computeGradient(self, X, y):
        prob = self.computeProb(X)
        d = y - prob
        dw = -np.mean(np.dot(X.T, d))
        db = -np.mean(d)
        return (dw, db)

    def updateParams(self, grads, lr):
        dw, db = grads
        self.w -= lr * dw
        self.b -= lr * db

    def computeNorm(self, a):
        return np.sqrt(np.sum(a * a))

    def fit(self, X, y, lr=1e-1, n_iter=300, epsilon=1e-7, n_patience=10):
        cost_history = []
        accuracy_history = []
        for i in range(n_iter):
            cost = self.computeCost(X, y)
            acc = (self.predict(X) == y).mean()
            cost_history.append(cost)
            accuracy_history.append(acc)
            grads = self.computeGradient(X, y)
            self.updateParams(grads, lr)
            if self.computeNorm(grads[0]) < epsilon:
                n_patience -= 1
            if n_patience == 0:
                break

        return np.array(cost_history), np.array(accuracy_history)
