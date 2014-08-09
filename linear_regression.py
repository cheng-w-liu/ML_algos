import numpy as np
import scipy.optimize as sci_opt


class Linear_Regression :

    """
    Note: any column vector will be represented as a 
             1d array (row vector with shape (m,))
                rather than (m,1) array
    """

    
    def computeCost(self, theta, X, y, lambda_reg = 0.0) :
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == theta.shape[0]
        m = X.shape[0]

        hX_y = np.zeros(m,'float')
        hX_y = X.dot(theta) - y
        J0 = (1.0/(2.0*m)) * sum(np.power(hX_y, 2.0))

        theta1 = np.array(theta); theta1[0] = 0.0
        regular = (lambda_reg/(2.0*m)) * sum(np.power(theta1, 2.0))
        
        J = J0 + regular
        return J
    # end of computeCost


    def computeGradient(self, theta, X, y, lambda_reg = 0.0) :
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == theta.shape[0]
        m = X.shape[0]

        grad = np.zeros_like(theta)
        theta1 = np.array(theta); theta1[0] = 0.0;
        hX_y = np.zeros(m,'float')
        hX_y = X.dot(theta) - y        
        grad = (1.0/m) * X.T.dot(hX_y) + (lambda_reg/m) * theta1
        return grad
    # end of computeGradient


    def trainLinearReg(self, X, y, lambda_reg = 0.0) :
        initial_theta = np.zeros(X.shape[1], 'float')
        
        opt_theta = sci_opt.fmin_cg(f=self.computeCost, x0=initial_theta,
                                    fprime=self.computeGradient,
                                    args=(X, y, lambda_reg),
                                    maxiter = 400,
                                    )        
        return opt_theta
    # end of trainLinearReg


    def learningCurve(self, X, y, Xval, yval, lambda_reg = 0.0) :
        m = X.shape[0]
        train_errors = np.zeros(m+1, 'float'); train_errors[0:2] = 0.0
        validate_errors = np.zeros(m+1, 'float'); validate_errors[0:2] = 0.0

        for i in range(2,m) :
            theta_trained = self.trainLinearReg(X[0:i,:], y[0:i], lambda_reg)
            train_errors[i] = self.computeCost(theta_trained, X[0:i,:], y[0:i], 0.0)
            validate_errors[i] = self.computeCost(theta_trained, Xval, yval, 0.0)
        # end of i

        return train_errors, validate_errors
    # end of learningCurve


    def polyFeatures(self, X0, p) :
        m = X0.shape[0]
        X = np.array(X0)
        for d in range(2,p+1,1) :
            X = np.hstack( (X, np.power(X[:,1].reshape(m,1), d) ) )
        # end of d
        return X
    # end of polyFeatures


    def featureNormalize(self, X) :
        normX = np.array(X, 'float')
        nu = np.zeros(X.shape[1], 'float')
        sigma = np.zeros(X.shape[1], 'float')
        for c in range(1, X.shape[1], 1) :
            nu[c] = np.mean(X[:,c])
            sigma[c] = np.std(X[:,c])
            normX[:,c] = (X[:,c]-nu[c])/sigma[c]
        # end of c
        return normX, nu, sigma
    # end of featureNormalize


    def featureNormalize_given(self, X, nu, sigma) :
        assert X.shape[1] == nu.shape[0]
        assert X.shape[1] == sigma.shape[0]
        normX = np.array(X, 'float')
        for c in range(1, X.shape[1], 1) :
            normX[:,c] = (X[:,c]-nu[c])/sigma[c]
        # end of c
        return normX
    # end of featureNormalize_given


    def validateCurve(self, X, y, Xval, yval) :
        lambda_vec = np.array([0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
        train_errors = np.zeros_like(lambda_vec)
        validate_errors = np.zeros_like(lambda_vec)
        for i in range(len(lambda_vec)) :
            lambda_reg = lambda_vec[i]
            trained_theta = self.trainLinearReg(X, y, lambda_reg)

            train_errors[i] = self.computeCost(trained_theta, X, y, 0.0)
            validate_errors[i] = self.computeCost(trained_theta, Xval, yval, 0.0)

        # end of for

        return lambda_vec, train_errors, validate_errors
    # end of validateCurve

# end of class Linear_Regression
