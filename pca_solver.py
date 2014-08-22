import numpy as np

class PCA_solver :
    
    def featureNormalize(self, rawX) :
        m, n = rawX.shape
        normX = np.zeros_like(rawX)
        mu = np.zeros(n, 'float')
        sigma = np.zeros(n, 'float')
        for c in range(n) :
            mu[c] = np.mean(rawX[:,c])
            sigma[c] = np.std(rawX[:,c])
            normX[:,c] = ( rawX[:,c] - mu[c] )/sigma[c]
        # end if c
        return normX, mu, sigma
    # end of featureNormalize


    def runPCA(self, X) :
        m = X.shape[0]
        Sigma = (1.0/m) * X.T.dot(X)
        U, S, V = np.linalg.svd(Sigma)
        return U, S
    # end of runPCA


    def projectData(self, X, U, K) :
        m, n = X.shape
        assert K <= n
        assert X.shape[1] == U.shape[0]
        Z = np.zeros((m, K), 'float')
        U_reduce = np.array(U[:,0:K])
        Z = X.dot(U_reduce)
        return Z
    # end of projectData


    def recoverData(self, Z, U, K) :
        assert Z.shape[1] == K
        U_reduce = np.array(U[:,0:K])
        X_rec = Z.dot(U_reduce.T)
        return X_rec
    # end of recoverData
        
# end of class PCA_solver


