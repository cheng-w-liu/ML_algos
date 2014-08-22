import numpy as np

class Newral_Network :

    def nnComputeCost(self, nn_params, S1, S2, S3, 
                      X, y, lambda_reg) :
        assert len(nn_params) == (S2*(S1+1) + S3*(S2+1))
        assert S3 == max(y)+1

        Theta1 = np.reshape( nn_params[0:S2*(S1+1)], (S2, S1+1) ) # S2*(S1+1) array
        Theta2 = np.reshape( nn_params[S2*(S1+1):] , (S3, S2+1) ) # S3*(S2+1) array
        m = X.shape[0]
        K = S3        
        Y = np.zeros((m,K), dtype='float')
        for t in range(m) : Y[t,y[t]] = 1

        A1 = np.hstack( (np.ones((m,1), dtype='float'), X) ) # m*(S1+1) array
        
        Z2 = A1.dot(Theta1.T) # m*S2 array
        A2 = np.hstack( (np.ones((m,1), dtype='float'), self.sigmoid(Z2)) ) # m*(S2+1) array

        Z3 = A2.dot(Theta2.T) # m*S3 array
        A3 = self.sigmoid(Z3) # m*S3 array

        J0_matrix = Y * np.log(A3) + (1.0-Y) * np.log(1.0 - A3)
        J0 = (-1.0/m) * sum( sum(J0_matrix, 0) )

        regular1 = sum( sum( np.power(Theta1[:,1:], 2.0) ) )
        regular2 = sum( sum( np.power(Theta2[:,1:], 2.0) ) )
        regular = (lambda_reg/(2.0*m)) * (regular1 + regular2)

        J = J0 + regular

        return J
    # end of nnComputeCost

    
    
    def nnComputeGradient(self, nn_params, S1, S2, S3, 
                            X, y, lambda_reg) :
        assert len(nn_params) == (S2*(S1+1) + S3*(S2+1))

        Theta1 = np.reshape( nn_params[0:S2*(S1+1)], (S2, S1+1) ) # S2*(S1+1) array
        Theta2 = np.reshape( nn_params[S2*(S1+1):] , (S3, S2+1) ) # S3*(S2+1) array
        m = X.shape[0]
        K = S3        

        Delta1 = np.zeros_like(Theta1) # S2*(S1+1) array
        Delta2 = np.zeros_like(Theta2) # S3*(S2+1) array

        delta2_tmp = np.zeros((S2+1,1), dtype='float')
        delta2 = np.zeros((S2,1), dtype='float')
        delta3 = np.zeros((S3,1), dtype='float')

        for t in range(m) :
            # --------------- Forward propagation --------------- #
            xi = X[t,:]
            xi = np.reshape(xi, (S1,1)) # S1*1 array
            a1 = np.vstack( (1,xi) ) # (S1+1)*1 array

            z2 = Theta1.dot(a1) # S2*1 array
            a2 = np.vstack( (1, self.sigmoid(z2)) ) # (S2+1)*1 array

            z3 = Theta2.dot(a2) # S3*1 array
            a3 = self.sigmoid(z3) # S3*1 array

            yi = np.zeros((K,1), dtype='float') # K*1 array, K = S3
            yi[y[t]] = 1.0

            delta3 = a3 - yi # S3*1 array
            
            # --------------- backward propagation --------------- #            
            Delta2 = Delta2 + delta3.dot(a2.T) # S3*(S2+1)

            delta2_tmp = (Theta2.T).dot(delta3) # (S2+1)*1 array
            delta2 = delta2_tmp[1:] * self.sigmoidGradient(z2) # S2*1 array
            Delta1 = Delta1 + delta2.dot(a1.T) # S2*(S1+1) array                        
        # end of t

        mask1 = np.ones_like(Theta1); mask1[:,0] = 0.0
        mask2 = np.ones_like(Theta2); mask2[:,0] = 0.0
        Delta1 = (1.0/m) * Delta1 + (lambda_reg/m) * mask1 * Delta1
        Delta2 = (1.0/m) * Delta2 + (lambda_reg/m) * mask2 * Delta2

        grad = np.hstack( (Delta1.ravel(), Delta2.ravel()) )
        return grad

    # end of nnComputeGradient



    def sigmoidGradient(self, z) :
        return self.sigmoid(z) * (1.0 - self.sigmoid(z))
    # end of sigmoidGradient



    def sigmoid(self, z) :
        return 1.0/( 1.0 + np.exp(-z) )
    # end of sigmoid



    def randomInitialization(self, L_in, L_out) :
        epsilon0 = 0.12
        W = np.random.random((L_out, L_in+1)) * 2 * epsilon0 - epsilon0
        return W
    # end of randomInitialization



    def predict(self, Theta1, Theta2, X) :
        m = X.shape[0]

        A1 = np.hstack( (np.ones((m,1), dtype='float'), X ) )

        Z2 = A1.dot(Theta1.T)
        A2 = self.sigmoid(Z2)
        A2 = np.hstack( (np.ones((m,1), dtype='float'), A2) )

        Z3 = A2.dot(Theta2.T)
        A3 = self.sigmoid(Z3)

        p = np.argmax(A3, 1)
        return p
    # end of predict

# end of class Neural_Network
