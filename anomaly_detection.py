import numpy as np

class AnomalyDetection :
    
    def estimateGaussian(self, X) :
        n = X.shape[1]
        mu = np.zeros(n, 'float')
        var = np.zeros(n, 'float')
        for c in range(n) :
            mu[c] = np.mean(X[:,c])
            var[c] = np.var(X[:,c]) #, ddof=1)
        return mu, var
    # end of estimateGaussian

    
    def multivariateGaussian(self, X, mu, var) :
        m, n = X.shape
        assert n == len(mu) and n == len(var)
        p = np.zeros(m, 'float')
        for r in range(m) :
            xi = X[r,:]
            p[r] = self.ProbProduct(xi, mu, var)
        return p
    # end of multivariateGaussian


    def ProbProduct(self, xi, mu, var) :
        probability = 1.0
        for j in range(len(xi)) : 
            probability *= self.singleProb(xi[j], mu[j], var[j])
        return probability
    # end of multivariateProb


    def singleProb(self, xj, muj, varj) :
        return (1.0/np.sqrt(2.0*np.pi*varj))*np.exp(-np.power(xj-muj,2.0)/(2.0*varj))
    # end of singleProb


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
