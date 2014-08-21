import numpy as np

class AnomalyDetection :

    def multivariateGaussian(self, X, mu, Sigma) :
        m, n = X.shape
        p = np.zeros(m, 'float')
        D = X - np.tile(mu, (m,1))
        invSigma = np.linalg.pinv(Sigma)
        f = ( np.power(2.0*np.pi, float(n)/2.0) ) * np.power( np.linalg.det(Sigma), 0.5 )        
        for i in range(m) :
            p[i] = (1.0/f) * np.exp( -0.5 * (D[i,:].dot(invSigma)).dot(D[i,:]) )
        # end of i
        return p
    # end of multivairateGaussian


    def computeMultivariateGaussian(self, xi, mu, Sigma) :
        n = len(xi)
        d = xi - mu
        invSigma = np.linalg.pinv(Sigma)
        f = ( np.power(2.0*np.pi, float(n)/2.0) ) * np.power( np.linalg.det(Sigma), 0.5 )        
        p = (1.0/f) * np.exp( -0.5 * (d.dot(invSigma)).dot(d) )
        return p
    # end of multivairateGaussian

    
    def estimateGaussian(self, X) :
        m = X.shape[0]
        mu = np.mean(X, 0)
        D = X - np.tile(mu, (m,1))
        Sigma = (1.0/float(m)) * D.T.dot(D)
        
        return mu, Sigma
    # end of estimateGaussian



    def selectThreshold(self, yval, pval) :
        bestEpsilon = 0.0
        bestF1 = 0.0
        Ngrid = 1000
        min_epsilon = max( min(pval), np.finfo('float').resolution )
        print 'min_eps : {0:}, max_eps : {1:}'.format(min_epsilon, max(pval))
        delta = (max(pval)-min_epsilon)/float(Ngrid)
        for i in range(Ngrid+1) :
            epsilon = min_epsilon + float(i) * delta
            predictions = np.less(pval, epsilon)
            tp = np.sum(np.logical_and(predictions == 1, yval == 1))
            fn = np.sum(np.logical_and(predictions == 0, yval == 1))
            fp = np.sum(np.logical_and(predictions == 1, yval == 0))
            pre = float(tp)/float(tp+fp)
            rec = float(tp)/float(tp+fn)
            F1 = 2.0*pre*rec/(pre+rec)
            if F1 > bestF1 :
                bestF1 = F1
                bestEpsilon = epsilon
        # end of i
        return bestEpsilon, bestF1
    # end of selectThreshold



# end of class AnomalyDetection
