import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops


class NeuralNetworkTF(object):
    
    def __init__(self, layers_dims):
        """
        Arguments:
            layers_dims -- lenght should be L+1
                           layers_dims[0] = dimension of the input = nx
                           layers_dims[l] = dimension of layer l, l = 1, ..., L        
        """
        self.layers_dims = np.copy(layers_dims)
        self.L = len(layers_dims) - 1
        
        
    def initializeParameters(self):
        """
        Return:
            a dictioanry of parameters: W1, b1, W2, b2, ..., WL, bL
        """
        
        parameters = {}
        layers_dims = self.layers_dims
        for l in range(1, self.L+1):
            parameters['W'+str(l)] = tf.get_variable(name='W'+str(l), dtype=tf.float64, 
                                                     shape=[layers_dims[l], layers_dims[l-1]], 
                                                     initializer=tf.contrib.layers.xavier_initializer())
            parameters['b'+str(l)] = tf.get_variable(name='b'+str(l), dtype=tf.float64,
                                                     shape=[layers_dims[l], 1], 
                                                     initializer=tf.zeros_initializer())
        return parameters

        
    def createPlaceHolder(self, n_x, n_y):
        """
        Arguments:
            n_x -- size of the input feature
            n_y -- size of the output feature
            
        Returns:
            X -- tf.placeholder object for the data matrix, of shape (n_x, None) and dtype "float"
            Y -- tf.placeholder object for the labels, of shape (n_y, None) and dtype "float"        
        """
        
        X = tf.placeholder(dtype=tf.float64, shape=(n_x, None), name='X')
        Y = tf.placeholder(dtype=tf.float64, shape=(n_y, None), name='Y')
        return X, Y
        
    
    def generateRandomMiniBatches(self, X, Y, mini_batch_size=64, seed=0):
        """
        Creates a list of random minibatches from (X, Y)
    
        Arguments:
            X -- input data, of shape (n_x, m) = (input size, number of examples)
            Y -- label vector, of shape (n_y, m) = (label size, number of examples)
            mini_batch_size -- size of the mini-batch, integer
    
        Returns:
            mini_batches -- list of (mini_batch_X, mini_batch_Y)
        """
        
        np.random.seed(seed)
        m = X.shape[1]
        num_complete_minibatches = m // mini_batch_size
        
        # 1. shuffle X, Y first
        permutation_idx = list(np.random.permutation(m))
        shuffled_X = X[:, permutation_idx]
        shuffled_Y = Y[:, permutation_idx]        
        
        mini_batches = []
        for k in range(num_complete_minibatches):
            start_idx = k * mini_batch_size
            end_idx = (k+1) * mini_batch_size
            mini_batch_X = X[:, start_idx: end_idx]
            mini_batch_Y = Y[:, start_idx: end_idx]
            mini_batches.append((mini_batch_X, mini_batch_Y))
            
        if m % mini_batch_size != 0:
            start_idx = num_complete_minibatches * mini_batch_size
            end_idx = m
            mini_batch_X = X[:, start_idx: end_idx]
            mini_batch_Y = Y[:, start_idx: end_idx]
            mini_batches.append((mini_batch_X, mini_batch_Y))
    
        return mini_batches
    
    
    def forwardPropagation(self, X, parameters):        
        """
        Perform forward propagation to get the linear output of the final layer
        All the hidden layers will be ReLU
        
        Arguments:
            X: input data, of shape (n_x, m) = (input size, number of examples)
            parameters: W1, b1, W2, b2, ..., WL, bL
            
        Returns:
            ZL: the linear output of the final layer, to be fed into the activation 
                function of the final layer 
        """
        L = self.L
        A_prev = X
        for l in range(1, L):
            W = parameters['W'+str(l)]
            b = parameters['b'+str(l)]
            Z = tf.add(tf.matmul(W, A_prev), b)
            A = tf.nn.relu(Z)
            A_prev = A
        
        W = parameters['W'+str(L)]
        b = parameters['b'+str(L)]
        ZL = tf.add(tf.matmul(W, A_prev), b)
        return ZL   
    
    
    def computeCost(self, ZL, Y):
        """
        Arguments:
            ZL -- n_y x m, linear units in the final layers, to be fed into the activation function
            Y -- n_y x m, the true label vector
            parameters -- a dictionary of parameter mapping
                          W1, b1, W2, b2, ...WL, bL
            lambd: the regularization constant                          
            
        Returns:
            cost function = cross entropy
        """        
     
        logits = tf.transpose(ZL)
        labels = tf.transpose(Y)
    
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    
        return cost    

    
    def fit(self, X_train, Y_train, learning_rate=0.01, num_epochs=1000, mini_batch_size=32):
        """
        Arguments:
            X -- input data, of shape (n_x, m)
            Y -- input labels, of shape (n_y, m)
            learning_rate -- learning rate of the optimization algorithm
            num_epochs -- number of epochs, one epoch corresponds to scanning through the entire data set once
            minibatch_size -- size of a minibatch                        
        
        Returns:
            costs -- the training cost after each of the epochs
        """
        
        ops.reset_default_graph()
        seed = 1
        
        # initialize parameters
        n_x, m = X_train.shape
        n_y = Y_train.shape[0]
        num_mini_batches = m // mini_batch_size
        
        parameters = self.initializeParameters()
        X, Y = self.createPlaceHolder(n_x, n_y)
    
        # specify the computation graph
        ZL = self.forwardPropagation(X, parameters)
        cost = self.computeCost(ZL, Y)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        
        init = tf.global_variables_initializer()
        
        costs = []

        with tf.Session() as sess:
            
            sess.run(init)
            
            for epoch in range(num_epochs):
                epoch_cost = 0.0
                num_mini_batches = m // mini_batch_size
                mini_batches = self.generateRandomMiniBatches(X_train, Y_train, mini_batch_size, seed=seed)
                seed += 1
                
                for mini_batch in mini_batches:
                    mini_batch_X, mini_batch_Y = mini_batch
                    
                    _ , mini_batch_cost = sess.run([optimizer, cost], 
                                                  feed_dict={X: mini_batch_X, Y: mini_batch_Y})
                    epoch_cost += mini_batch_cost / num_mini_batches
                    
                costs.append(epoch_cost)
                
            self.parameters = sess.run(parameters)
            
            writer = tf.summary.FileWriter("./graphs", sess.graph)            
            
        return costs
            
        
    def predict_probas(self, X):
        ZL = self.forwardPropagation(X, self.parameters)
        AL = tf.nn.softmax(logits=ZL, dim=0)
        
        with tf.Session() as sess:
            probas = sess.run(AL)
            
        return probas
        

