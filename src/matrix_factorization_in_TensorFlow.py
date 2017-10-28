import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops


class MatrixFactorizationTF(object):

    def __init__(self, K):
        """
        Arguments:
            K -- number of hidden dimension
        """
        self.K = K


    def initializeParameters(self, m, n):
        """
        Arguments:
            m -- number of users
            n -- number of items

        Returns:
            parameters -- parameters['b'], global bias, scalar
                          parameters['u'], users bias, shape (m, 1)
                          parameters['d'], item bias, shape (1, n)
                          parameters['P'], users feature matrix, shape (m, K)
                          parameters['Q'], items feature matrix, shape (n, K)        
        """
        k = self.K
        
        parameters = {}
        parameters['b'] = tf.get_variable(name='b', dtype=tf.float64, shape=[],
                                          initializer=tf.zeros_initializer())

        parameters['u'] = tf.get_variable(name='u', dtype=tf.float64, shape=[m, 1],
                                          initializer=tf.zeros_initializer())

        parameters['d'] = tf.get_variable(name='d', dtype=tf.float64, shape=[1, n],
                                          initializer=tf.zeros_initializer())

        parameters['P'] = tf.get_variable(name='P', dtype=tf.float64, shape=[m, k],
                                          initializer=tf.random_normal_initializer())

        parameters['Q'] = tf.get_variable(name='Q', dtype=tf.float64, shape=[n, k],
                                          initializer=tf.random_normal_initializer())

        return parameters


    def createPlaceHolder(self, m, n):
        """
        Arguments:
            m -- number of users
            n -- number of items

        Returns:
            A placeholder for the rating matrix for training
        """
        Y = tf.placeholder(dtype=tf.float64, shape=(m, n), name='Y')
        return Y
    
    
    def predict(self, parameters):
        """
        Arguments:
            parameters -- parameters['b'], global bias, scalar
                          parameters['u'], users bias, shape (m, 1)
                          parameters['d'], item bias, shape (1, n)
                          parameters['P'], users feature matrix, shape (m, K)
                          parameters['Q'], items feature matrix, shape (n, K)                         
        
        Returns:
            Y_hat -- the predicted rating matrix, shape (m, n)            
        """

        b = parameters['b']
        u = parameters['u']
        d = parameters['d']
        P = parameters['P']
        Q = parameters['Q']

        Y_hat = b + u + d + tf.matmul(P, tf.transpose(Q))
        return Y_hat
        
    
    def computeCost(self, Y_hat, Y, parameters, beta=0.0):
        """
        Arguments:
            Y_hat -- the predicted rating matrix, shape (m, n)
            Y -- the ground truth rating matrix, shape (m, n)
            parameters -- parameters['b'], global bias, scalar
                          parameters['u'], users bias, shape (m, 1)
                          parameters['d'], item bias, shape (1, n)
                          parameters['P'], users feature matrix, shape (m, K)
                          parameters['Q'], items feature matrix, shape (n, K)                         
            beta -- regularization term

        Returns:
            J -- the cost function
        """

        R = tf.cast(Y > 0, dtype=tf.float64)
        L = tf.reduce_sum(R)

        J0 = (1.0/(2.0 *L)) * tf.reduce_sum(tf.pow( tf.multiply(Y_hat - Y, R) , 2))

        u = parameters['u']
        d = parameters['d']
        P = parameters['P']
        Q = parameters['Q']
        
        u2 = tf.pow(tf.norm(u), 2)
        d2 = tf.pow(tf.norm(d), 2)
        P2 = tf.pow(tf.norm(P), 2)
        Q2 = tf.pow(tf.norm(Q), 2)

        if beta > 0.0:
            reg = (beta/(2.0*L)) * ( u2 + d2 + P2 + Q2 )
        else:
            reg = 0.0

        J = J0 + reg
        return J

    
    def fit(self, Y, alpha=0.01, beta=1.0, num_iters=1000):
        """
        Arguments:
            Y -- ground truth rating matrix, shape (m, n)
            alpha -- learning rate
            beta -- regularization term
            num_iter -- number of iterations

        Returns:
            costs -- cost during training
        """

        ops.reset_default_graph()
        m, n = Y.shape
        parameters = self.initializeParameters(m, n)
        placeholder_Y = self.createPlaceHolder(m, n)

        Y_hat = self.predict(parameters)
        cost = self.computeCost(Y_hat, placeholder_Y, parameters, beta=0.0)
        optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(cost)

        init = tf.global_variables_initializer()

        costs = []

        with tf.Session() as ss:
            
            ss.run(init)

            for i in range(num_iters):
                _, c = ss.run([optimizer, cost], feed_dict={placeholder_Y: Y})
                costs.append(c)

            self.parameters = ss.run(parameters)
            writer = tf.summary.FileWriter("./graphs", ss.graph)
            
        writer.close()
        
        return costs

    
    def predict_ratings(self):
        if not hasattr(self, 'parameters'):
            print('do not have parameters')
            return None

        Y_hat = self.predict(self.parameters)
        with tf.Session() as ss:
            ratings = ss.run(Y_hat)
            
        return ratings
