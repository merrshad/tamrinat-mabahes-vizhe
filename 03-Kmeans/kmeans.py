import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


np.random.seed(42)
scores = np.random.uniform(0, 20, 30).reshape(-1, 1)


kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto')
kmeans.fit(scores)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_


for i, (score, label) in enumerate(zip(scores.flatten(), labels)):
    print(f"Student {i+1}: Score = {score:.2f} → Cluster {label}")

print("\nAverage score of each cluster:")
for i, center in enumerate(centroids):
    print(f"Cluster {i}: {center[0]:.2f}")

 
cluster_0 = [i+1 for i, label in enumerate(labels) if label == 0]
cluster_1 = [i+1 for i, label in enumerate(labels) if label == 1]

print("\nStudents in each cluster:")
print(f"Cluster 0: {cluster_0}")
print(f"Cluster 1: {cluster_1}")
