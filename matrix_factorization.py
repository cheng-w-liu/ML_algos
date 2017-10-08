import numpy as np

class MatrixFactorization(object):
    
    def __init__(self, K=10):
        """
        Arguments:
            K -- dimension of the hidden variable        
        """
        self.K = K
    
    
    def fit(self, Y, num_iterations=1000, alpha=0.001, beta=0):
        """
        Arguments:
            Y -- M x N raring matrix,  
                 where M is the number of users, N the number of items
            alpha -- learning rate for the gradient descent
            beta -- regularization term
                 
            Y will be approximated by two matrices P and Q, plus biases b, u, and d, where
                b a global bias
                u a M x 1 matrix for users biases
                d a 1 x N matrix for iterm biases            
                P a M x K matrix
                Q a N x K matrix
            Each row vector of P is the feature vector for a given user
            Each row vector of Q is the feature vector for a given item
            Y_hat = b + u + d + P Q^T                   
            trained parameters will be saved to self.parameters.            
            
        Returns:
            costs: cost during the training phase
            
            
        """
        M, N = Y.shape
        parameters = self.initializeParameters(M, N)
        costs = []
        for i in range(num_iterations):
            if i % 10 == 0:
                costs.append(self.computeCost(Y, parameters, beta))
            gradients = self.computeGradients(Y, parameters, beta)
            parameters = self.updateParameters(parameters, gradients, alpha)
        self.parameters = parameters    
        return costs
            
 
    def initializeParameters(self, M, N):
        """
        arguments:
            M -- number of users, for P matrix
            N -- number of items, for Q matrix
        
        returns:
            parameters -- a dictionary of b, u, d, P, Q
            parameters['b'] -- scalar, global bias
            parameters['u'] -- M x 1, users biases
            parameters['d'] -- 1 x N, item biases            
            parameters['P'] -- M x K matrix, whose rows correspond to the feature for each user
            parameters['Q'] -- N x K matrix, whose rows correspond to the feature for each item
        """
        parameters = {}        
        K = self.K
        parameters['b'] = 0.0
        parameters['u'] = np.zeros((M, 1))    
        parameters['d'] = np.zeros((1, N))        
        parameters['P'] = np.random.randn(M, K) * np.sqrt(2.0/K)
        parameters['Q'] = np.random.randn(N, K) * np.sqrt(2.0/K)
        return parameters


    def updateParameters(self, parameters, gradients, alpha):
        """
        arguments:
            parameters -- b, u, d, P, Q
            gradients -- db, du, dd, dP, dQ
            alpha -- learning rate
        
        returns:
            parameters -- updated parameters based on the gradients
        """
        parameters['b'] = parameters['b'] - alpha * gradients['db']
        parameters['u'] = parameters['u'] - alpha * gradients['du']
        parameters['d'] = parameters['d'] - alpha * gradients['dd']                
        parameters['P'] = parameters['P'] - alpha * gradients['dP']
        parameters['Q'] = parameters['Q'] - alpha * gradients['dQ']
        return parameters
                            
        
    def computeCost(self, Y, parameters, beta=0.0):
        """
        Arguments:
            Y -- the rating matrix, N x M
            parameters['b'] -- scalar, global bias
            parameters['u'] -- M x 1, users biases
            parameters['d'] -- 1 x N, item biases            
            parameters['P'] -- M x K matrix, whose rows correspond to the feature for each user
            parameters['Q'] -- N x K matrix, whose rows correspond to the feature for each item
            
            Y_hat = b + u + d + P Q^T
            beta -- regularization term
            
        Returns:
            cost -- J
            
        J = J0 + regularization
        
        J0 = 1/2L * \sum_{i,j : r_{ij}=1} ( y_{ij} - y_{ij}_hat ) ^ 2
        where y_{ij}_hat = b + u_i + d_j + p_i^T q_j

        regularization = beta/2L * (||P||_F^2 + ||Q||_F^2 + ||u||_F^2 + ||d||_F^2), 
        where ||...||_F^2 is the Forbenius norm
        """
        P = parameters['P']
        Q = parameters['Q']
        u = parameters['u']
        d = parameters['d']                
        R = (Y > 0).astype(np.float32)
        L = float(Y.shape[0] * Y.shape[1])
        Y_hat = self.predict(parameters)
        E = np.multiply(Y - Y_hat, R)
        J0 = (1.0/(2.0*L)) * np.sum(np.power(E, 2))
        reg = (beta/(2.0*L)) * ( np.linalg.norm(P) + np.linalg.norm(Q) + np.linalg.norm(u) + np.linalg.norm(d) )
        J = J0 + reg
        return J

    
    def computeGradients(self, Y, parameters, beta):
        """
        arguments:
            Y -- the rating matrix, N x M
            parameters['b'] -- scalar, global bias
            parameters['u'] -- M x 1, users biases
            parameters['d'] -- 1 x N, item biases            
            parameters['P'] -- M x K matrix, whose rows correspond to the feature for each user
            parameters['Q'] -- N x K matrix, whose rows correspond to the feature for each item
            beta -- regularization
            
        returns:
            derivative of Y w.r.t. b, u, d, P, Q respectively
            dJ/db = -1/L * np.sum(E * R)                       , scalar
            dJ/du = -1/L * np.sum(E * R, axis=1) + beta/L * u  , M x 1 matrix
            dJ/dd = -1/L * np.sum(E * R, axis=0) + beta/L * d  , 1 x N matrix            
            dJ/dP = -1/L * (E * R) Q + beta/L * P              , M x K matrix
            dJ/dQ = -1/L * (E * R)^T P + beta/L * Q            , N x K matrix
        """
        P = parameters['P']
        Q = parameters['Q']
        u = parameters['u']
        d = parameters['d']        
        R = (Y > 0).astype(np.float32)                               
        L = float(Y.shape[0] * Y.shape[1])
        Y_hat = self.predict(parameters)
        E = np.multiply(Y - Y_hat, R)
        
        gradients = {}        
        gradients['db'] = (-1.0/L) * np.sum(E)
        gradients['du'] = (-1.0/L) * (np.sum(E, axis=1, keepdims=True) - beta * u)
        gradients['dd'] = (-1.0/L) * (np.sum(E, axis=0, keepdims=True) - beta * d)
        gradients['dP'] = (-1.0/L) * (np.dot(E, Q) - beta * P)
        gradients['dQ'] = (-1.0/L) * (np.dot(E.T, P) - beta * Q)
        
        return gradients
    
    
    def predict(sefl, parameters):
        """
        arguments:
            parameters -- a dictionary for b, u, d, P, Q
            parameters['b'] -- scalar, global bias
            parameters['u'] -- M x 1, users biases
            parameters['d'] -- 1 x N, item biases            
            parameters['P'] -- M x K matrix, whose rows correspond to the feature for each user
            parameters['Q'] -- N x K matrix, whose rows correspond to the feature for each item
            
        returns:
            Y_hat = b + d + u + P Q^T
        """
        b = parameters['b']
        u = parameters['u']
        d = parameters['d']        
        P = parameters['P']
        Q = parameters['Q']
        Y_hat = b + u + d + np.dot(P, Q.T)    
        return Y_hat
        
              




