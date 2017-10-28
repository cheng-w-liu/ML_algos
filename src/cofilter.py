import numpy as np

class CollaborativeFiltering :

    
    def computeCost(self, params, Y, R, num_movies, num_users, num_features, lambda_reg) :
        X = np.reshape(params[0:num_movies*num_features], (num_movies, num_features) )
        Theta = np.reshape(params[num_movies*num_features:], (num_users, num_features) )
        RXThetat_Y = R * (X.dot(Theta.T) - Y)
        J0 = 0.5 * sum(sum(np.power(RXThetat_Y, 2.0)))
        regular1 = 0.5 * sum(sum(np.power(X, 2.0)))
        regular2 = 0.5 * sum(sum(np.power(Theta, 2.0)))
        J = J0 + lambda_reg * (regular1 + regular2)
        return J
    # end of computeCost


    def computeGradient(self, params, Y, R, num_movies, num_users, num_features, lambda_reg) :
        assert Y.shape[0] == num_movies and Y.shape[1] == num_users

        X = np.reshape(params[0:num_movies*num_features], (num_movies, num_features) )
        Theta = np.reshape(params[num_movies*num_features:], (num_users, num_features) )

        grad_X = np.zeros_like(X)
        grad_Theta = np.zeros_like(Theta)

        grad_X = ((X.dot(Theta.T) - Y) * R).dot(Theta) + lambda_reg * X
        grad_Theta = ((X.dot(Theta.T) - Y) * R).T.dot(X) + lambda_reg * Theta

        grad = np.hstack( (grad_X.ravel(), grad_Theta.ravel()) )
        return grad
    # end of computeGradient


    def meanNormalize(self, Y) :
        normY = np.array(Y)
        mu = np.mean(Y, axis=1)
        normY = Y - np.reshape(mu, (len(mu),1))
        return normY, mu
    # end of meanNormalize


    
# end of CollaborativeFiltering
