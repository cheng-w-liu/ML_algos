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
        for i in range(num_movies) :
            select = np.where( R[i,:] == 1 )
            idx = select[0]
            # for a given movie i, only consider those users who rated i
            Theta_select = Theta[idx, :] 
            Y_select = Y[i,idx]
            grad_X[i,:] = (X[i,:].dot(Theta_select.T) - Y_select).dot(Theta_select)
            grad_X[i,:] = grad_X[i,:] + lambda_reg * X[i,:]
        # end of i


        grad_Theta = np.zeros_like(Theta)
        for j in range(num_users) :
            select = np.where( R[:,j] == 1)
            idx = select[0]
            # for a given user j, only consider those movies rated by j
            X_select = X[idx, :]
            Y_select = Y[idx, j]
            grad_Theta[j,:] = (X_select.dot(Theta[j,:].T) - Y_select).T.dot(X_select)
            grad_Theta[j,:] = grad_Theta[j,:] + lambda_reg * Theta[j,:]
        # end of j

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
