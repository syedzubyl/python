import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_clusters, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters

    def fit(self, X):
       
        centroids_idx = np.random.choice(X.shape[0], size=self.n_clusters, replace=False)
        self.centroids = X[centroids_idx]

        for _ in range(self.max_iters):
           
            labels = self._assign_clusters(X)
           
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            
           
            if np.allclose(self.centroids, new_centroids):
                break
            
            self.centroids = new_centroids

    def _assign_clusters(self, X):
       
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
       
        return np.argmin(distances, axis=1)

if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(100, 2)

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)

    labels = kmeans._assign_clusters(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], marker='X', c='red', s=200, label='Centroids')
    plt.title('K-means Clustering')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()
 