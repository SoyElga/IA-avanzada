import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12379)

def euclidean_distance(x1,x2):
  return np.sqrt(np.sum(x1-x2)**2)


class KMeans:
  def __init__(self, K=2, max_iters = 1000, plot_steps=False):
    self.K = K
    self.max_iters = max_iters
    self.plot_steps = plot_steps

    self.clusters = [[] for i in range(self.K)]

    self.centroids = []
  
  def predict(self, X):
    self.X = X
    self.n_samples, self.n_features = X.shape

    random_sample_id = np.random.choice(self.n_samples, self.K, replace=False)
    self.centroids = [self.X[i] for i in random_sample_id]

    for i in range(self.max_iters):
      #Update clusters
      self.clusters = self._create_clusters(self.centroids)
      if self.plot_steps:
        self.plot()

      #Update centroids
      centroids_old = self.centroids
      self.centroids = self._get_centroids(self.clusters)

      if self.plot_steps:
        self.plot()

      #Check if converged
      if self._is_converged(centroids_old, self.centroids):
        break

    #return cluster_lables
    return self._get_cluster_labels(self.clusters)

  def _get_cluster_labels(self, clusters):
    labels = np.empty(self.n_samples)
    for cluster_id, cluster in enumerate(clusters):
      for sample_id in cluster:
        labels[sample_id] = cluster_id
    return labels
      
  
  def _create_clusters(self, centroids):
    clusters = [[] for k in range(self.K)]
    for j, sample in enumerate(self.X):
      centroid_id = self._closest_centroid(sample, centroids)
      clusters[centroid_id].append(j)
    return clusters
  
  def _closest_centroid(self, sample, centroids):
    distances = [euclidean_distance(sample, point) for point in centroids]
    closest_id = np.argmin(distances)
    return closest_id

  def _get_centroids(self, clusters):
    centroids = np.zeros((self.K, self.n_features))
    for cluster_id, cluster in enumerate(clusters):
      cluster_mean = np.mean(self.X[cluster], axis=0)
      centroids[cluster_id] = cluster_mean
    return centroids

  def _is_converged(self, centroids_old, centroids):
    distances = [euclidean_distance(centroids_old[j], centroids[j]) for j in range(self.K)]
    return sum(distances) == 0
  
  def plot(self):
    fig, ax = plt.subplots(figsize=(12,8))

    for i, index in enumerate(self.clusters):
      point = self.X[index].T
      ax.scatter(*point)

    for point in self.centroids:
      ax.scatter(*point, marker="x", color="black", linewidth=2)

    plt.show()
