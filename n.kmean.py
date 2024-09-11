import numpy as np
import matplotlib.pyplot as plt
data=np.array([
        [1,2],
        [1,4],
        [1,0],
        [10,2],
        [10,4],
        [10,0]
        ])
plt.scatter(data[:,0],data[:,1],s=100)
plt.title('dataset')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.show()
def k_means_simple(data,k,max_iters=100):
     np.random.seed(42)
     centroids=data[np.random.choice(data.shape[0],k,replace=false)]
     for _ in range(max_iters):
         distance=np.linalg.norm(data[:,np.newaxis]-centroids,axis=2)
         cluster_labels=np.argmin(distance,axis=1)
         new.centroids=np.array([data[cluster_labels==i].mean(axis=0)for i in range(k)])
     if np.all(centroids==new_centroids):
         break
     centroids=new_centroids
     return centroids,cluster_labels
   k=2
   centroids,cluster_labels=k_meana_simple(data,k)
   plt.scatter(data[:,0],data[:,1].c=cluster label,s=100,cmap='virdis')
   plt.scatter(centroids[:,0],centroids[:,1],c='red',s=200,marker='x')
   plt.title('k-means clustering')
   plt.xlabel('feature 1')
   plt.ylabel('feature 2')
   plt.show()