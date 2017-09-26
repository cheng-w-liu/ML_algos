import numpy as np

class NeuralNetwork(object):
    
    def __init__(self, layers_dims):
        """        
        Return:
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
            parameters['W'+str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2.0/layers_dims[l-1])
            parameters['b'+str(l)] = np.zeros((layers_dims[l], 1))    
            
        return parameters
 

    def forwardPropagation(self, X, parameters):
        """        
        Arguments:
            X -- nx x m, each column corresponds to a data point
            parameters -- W1, b1, W2, b2, ...WL, bL, output of initializeParameters
            
        Return:
            AL -- output of the last layer L
            caches -- list of caches, each element corresponds to 
                      (linear_cache, activation_cache), where
                         linear_cache = (A_^[l-1], W^[l], b^[l])
                         activation_cache = Z^[l]                         
                     *Note that caches[l-1] corresponds to layer l
        """
        L = self.L
        caches = []
        A_prev = X
        for l in range(1, self.L):
            W = parameters['W'+str(l)]
            b = parameters['b'+str(l)]
            A, cache = self.linearActivationForward(A_prev, W, b, 'relu')
            caches.append(cache)
            A_prev = A
        
        W = parameters['W'+str(L)]
        b = parameters['b'+str(L)]
        AL, cache = self.linearActivationForward(A_prev, W, b, 'sigmoid')
        caches.append(cache)
        
        assert(AL.shape == (1,X.shape[1]))
        
        return AL, caches

    
    def linearActivationForward(self, A_prev, W, b, activation):
        """
        Arguments:
            A_prev -- A^[l-1], activation from the previous layer            
            W -- W^[l], weights of layer-l, n^[l] x n^[l-1]
            b -- b^[l], bias of layer-l, n^[l] x 1
            
        Return:
            Z -- Z^[l]
            linear_cache -- (A_prev, W, b), as defined above 
        """
        
        assert activation in ['relu','sigmoid']
        
        if activation == 'relu':
            Z, linear_cache = self.linearForward(A_prev, W, b)
            A, activation_cache = self._relu(Z)
        elif activation == 'sigmoid':
            Z, linear_cache = self.linearForward(A_prev, W, b)
            A, activation_cache = self._sigmoid(Z)
            
        cache = (linear_cache, activation_cache)
        return A, cache
    
    
    def linearForward(self, A_prev, W, b):
        """
        Arguments:
            A_prev -- A^[l-1], activation from the previous layer   
            W -- W^[l], weights of layer-l, n^[l] x n^[l-1]
            b -- w^[l], bias of layer-l, n^[l] x 1
        
        Returns:
            Z -- Z^[l]
            cache -- (A_prev, W, b)
        """
        
        Z = np.dot(W, A_prev) + b
        
        assert(Z.shape == (W.shape[0], A_prev.shape[1]))
        cache = (A_prev, W, b)
    
        return Z, cache 

    
    def backwardPropagation(self, AL, Y, caches):
        """
        Arguments:
            AL -- 1 x m, probability vector, output from the layer-L
            Y -- 1 x m, the true label vector
            caches -- list of caches, each element corresponds to 
                      (linear_cache, activation_cache), where
                         linear_cache = (A_^[l-1], W^[l], b^[l])
                         activation_cache = Z^[l]                         
                     *Note that caches[l-1] corresponds to layer l
                     
        Returns:
            grads -- a dictionary with gradients
                      grads['dW' + str(l)]
                      grads['db' + str(l)] 
                      grads['dA' + str(l)]            
        """
        
        grads = {}
        L = len(caches)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        
        current_cache = caches[L-1]
        dA_prev_temp, dW_temp, db_temp = self.linearActivationBackward(dAL, current_cache, 'sigmoid')
        grads['dA'+str(L-1)] = dA_prev_temp
        grads['dW'+str(L)] = dW_temp
        grads['db'+str(L)] = db_temp
        
        for l in range(L-1, 0, -1):
            current_cache = caches[l-1]
            dA = grads['dA'+str(l)]
            dA_prev_temp, dW_temp, db_temp = self.linearActivationBackward(dA, current_cache, 'relu')
            grads['dA'+str(l-1)] = dA_prev_temp
            grads['dW'+str(l)] = dW_temp
            grads['db'+str(l)] = db_temp
                    
        return grads
    
    
    def linearActivationBackward(self, dA, cache, activation):
        """
        Arguments:
            dA -- dA^[l], for layer-l, computed from the previous iteration at layer l+1
            cache -- tuple of (linear_cache, activation_cache)
                       where linear_cache = (A_prev, W, b)
                               A_prev = A^[l-1], 
                               W = W^[l], 
                               b = b^[l]
                             activation_cache = Z
                               Z = Z^[l]
            activation -- `sigmoid` or `relu`  
            
        Returns:
            dA_prev -- dA^[l-1]
            dW -- dW^[l]
            db -- db^[l]
        """
        
        assert activation in ['sigmoid', 'relu']
        
        linear_cache, activation_cache = cache
        if activation == 'sigmoid':
            dZ = self.sigmoidBackward(dA, activation_cache)
            dA_prev, dW, db = self.linearBackward(dZ, linear_cache)
            
        elif activation == 'relu':
            dZ = self.reluBackward(dA, activation_cache)
            dA_prev, dW, db = self.linearBackward(dZ, linear_cache)
            
        return dA_prev, dW, db    
    
    
    def linearBackward(self, dZ, linear_cache):
        """
        Arguments:
            dZ -- dZ^[l]
            linear_cache -- (A_prev, W, b), where
                               A_prev -- A^[l-1], activation from the previous layer            
                               W -- W^[l], weights of layer-l, n^[l] x n^[l-1]
                               b -- b^[l], bias of layer-l, n^[l] x 1
                               
        Returns:
            dA_prev -- dA^[l-1]
            dW -- dW^[l]
            db -- db^[l]            
        """
        
        A_prev, W, b = linear_cache
        m = A_prev.shape[1]
        dW = (1.0/m) * np.dot(dZ, A_prev.T)
        db = (1.0/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)            
        
        return dA_prev, dW, db

    
    def updateParameters(self, parameters, grads, learning_rate):
        L = len(parameters) // 2
        for l in range(1,L+1):
            parameters['W'+str(l)] = parameters['W'+str(l)] - learning_rate * grads['dW'+str(l)]
            parameters['b'+str(l)] = parameters['b'+str(l)] - learning_rate * grads['db'+str(l)]
        return parameters
    
    
    def computeCost(self, AL, Y, parameters, lambd):
        """
        Arguments:
            AL -- 1 x m, probability vector, output from the layer-L
            Y -- 1 x m, the true label vector
            parameters -- a dictionary of parameter mapping
                          W1, b1, W2, b2, ...WL, bL
            lambd: the regularization constant                          
            
        Returns:
            cost function = cross entropy + L2
        """
        
        L = len(parameters) // 2
        cross_entropy = self.computeCrossEntropy(AL, Y)
        
        L2_penalty = 0.0
        for l in range(1, L+1):
            W = parameters['W'+str(l)]
            L2_penalty += np.linalg.norm(W)
        L2_penalty *= lambd/(2.0 * m)  
        L2_penalty = np.squeeze(L2_penalty)
        
        cost = cross_entropy + L2_penalty
        
        return cross_entropy
    
    
    def computeCrossEntropy(self, AL, Y):
        """
        Arguments:
            AL -- 1 x m, probability vector, output from the layer-L
            Y -- 1 x m, the true label vector
            
        Return:
            cross-entropy
        """
        
        m = AL.shape[1]
        cross_entropy = (-1.0/float(m)) * ( np.sum(Y * np.log(AL)) + np.sum((1.0-Y) * np.log(1.0-AL)) )        
        cross_entropy = np.squeeze(cross_entropy)
        
        return cross_entropy
        
    
    def _sigmoid(self, Z):
        return 1.0/(1.0 + np.exp(-Z)), Z

    
    def _relu(self, Z):
        return np.maximum(Z , 0), Z

    
    def sigmoidBackward(self, dA, Z):
        #res = dA * self._sigmoid_derivative(Z)
        #return res
        return dA * self._sigmoid_derivative(Z)

    
    def reluBackward(self, dA, Z):
        return dA * self._relu_derivative(Z)
            
        
    def _sigmoid_derivative(self, Z):
        #derivative = self._sigmoid(Z) * (1.0 - self._sigmoid(Z))
        sigmoid_fun_values, _ = self._sigmoid(Z)
        derivative = sigmoid_fun_values * (1.0 - sigmoid_fun_values)
        return derivative  #, Z
    
    def _relu_derivative(self, Z):
        derivative = (Z > 0).astype(float)        
        return derivative #, Z
    
    
    def fit(self, X, Y, learning_rate=0.005, num_iterations=1000):
        """
        Arguments:
        X -- trainin data, nx x m
        Y -- true outcome, 1 x m
        
        Returns:
        history of training costs
        """
        
        costs = []        
        parameters = self.initializeParameters()
        for i in range(num_iterations):
            AL, caches = self.forwardPropagation(X, parameters)
            if i % 50 == 0:
                costs.append(self.computeCrossEntropy(AL, Y))
            grads = self.backwardPropagation(AL, Y, caches)
            parameters = self.updateParameters(parameters, grads, learning_rate)
        
        self.parameters = parameters
        return costs
    
    def predict_proba(self, X):
        AL, _ = self.forwardPropagation(X, self.parameters)
        return AL
    
    def predict(self, X):
        AL = self.predict_proba(X)
        Y_hat = (AL > 0.5).astype(float)
        return Y_hat
    

