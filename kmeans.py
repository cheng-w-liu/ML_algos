import numpy as np
import matplotlib.pyplot as plt

class kMeans :

    def plotData(self, X, idx, K, old_centroids, new_centroids) :
        assert X.shape[0] == idx.shape[0]
        assert K == len(new_centroids)
        assert len(old_centroids) == len(new_centroids)
        
        colors = ['r', 'g', 'b']
        markers = ['o', 's', 'v']
        xmin, xmax = 0.9 * min(X[:,0]), 1.1 * max(X[:,0])
        ymin, ymax = 0.9 * min(X[:,1]), 1.1 * max(X[:,1])
        plt.ion()
        plt.figure()
        for k in range(K) :
            subset = np.equal(idx, k)
            plt.plot(X[subset, 0], X[subset, 1], marker=markers[k], color=colors[k], 
                     markersize=8, linewidth=0)
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
            plt.xlabel('x coordinate')
            plt.ylabel('y coordinate')

            plt.plot(old_centroids[k,0], old_centroids[k,1], color='k', marker='x',
                     linewidth=0, markersize=15, markeredgewidth=5,
                     )

            plt.draw()
            plt.show()
        # end of k
    # end of plotData
    

    def runkMeans(self, X, K, max_iters = 10, progress = True) :
        m, n = X.shape
        #K = initial_centroids.shape[0]
        #centroids = np.array(initial_Centroids)

        centroids = self.kMeansInitCentroids(X, K)

        idx = np.zeros(m,'int_')
        for j in range(max_iters) :
            idx = self.findClosestCentroids(X, centroids)
            previous_centroids = centroids
            centroids = self.computeCentroids(X, idx, K)
            if progress is True :
                self.plotData(X, idx, K, previous_centroids, centroids)
            # end if
        # end of j
        return centroids, idx
    # end of runkMeans

    
    def kMeansInitCentroids(self, X, K) :
        n = X.shape[1]
        centroids = np.zeros((K, n), 'float')
        copyX = np.array(X)
        np.random.shuffle(copyX)
        centroids[:] = copyX[0:K]
        return centroids
    # end of kMeansInitCentroids


    def findClosestCentroids(self, X, centroids) :
        assert X.shape[1] == centroids.shape[1]
        K = centroids.shape[0]
        m = X.shape[0]
        idx = np.zeros(m,'int_')
        for i in range(m) :
            d_min = 1.0e+20
            ci = 0
            for k in range(K) :
                d = sum( np.power(X[i,:] - centroids[k,:], 2.0) )
                if d < d_min :
                    d_min = d
                    ci = k
                # end if
            # end of k
            idx[i] = ci
        # end of i
        return idx
    # end of findClosestCentroids


    def computeCentroids(self, X, idx, K) :
        assert X.shape[0] == idx.shape[0]
        n = X.shape[1]
        centroids = np.zeros((K,n), 'float')
        for k in range(K) :
            subset = np.equal(idx, k)
            centroids[k] = np.mean(X[subset,:], axis=0)
        # end of k
        return centroids
    # end of computeCentroids


# end of class kMeans
