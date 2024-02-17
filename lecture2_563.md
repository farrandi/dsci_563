# Lecture 2: DBSCAN and Hierarchical Clustering

## DBSCAN

- Density-Based Spatial Clustering of Applications with Noise
- **Idea**: Clusters are dense regions in the data space, separated by low-density regions
- Addresses K-Means' weaknesses:
  - No need to specify number of clusters
  - Can find clusters of arbitrary shapes
  - Can identify points that don't belong to any cluster

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X)

dbscan.labels_
```

- `eps` (default=0.5): maximum distance between two samples for one to be considered as in the neighborhood of the other
- `min_samples` (default = 5): number of samples in a neighborhood for a point to be considered as a core point

### How DBSCAN works

- **Kinds of points**:

  - **Core point**: A point that has at least `min_samples` points within `eps` of it
  - **Border point**: A point that is within `eps` of a core point, but has less than `min_samples` points within `eps` of it
  - **Noise point**: A point that is neither a core point nor a border point

- **Algorithm**:
  - randomly pick a point that has not been visited
  - Check if it's a core point
    - See `eps` distance around the point if there are `min_samples` points
  - If yes, start a cluster around this point
  - Check if neighbors are core points and repeat
  - Once no more core points, pick another random point and repeat

## Hierarchical Clustering

- Main idea:
  - Start with each point as a cluster
  - Merge the closest clusters
  - Repeat until only a single cluster remains (n-1 steps)
- Visualized as a `dendrogram`

```python
from matplotlib import pyplot as plt
from scipy.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, ward

X_scaled = StandardScaler().fit_transform(X)

linkage_array = ward(X_scaled)

# Plot the dendrogram
ax = plt.subplot()
dendrogram(linkage_array, ax=ax, color_threshold=3)
```

### Flat Clusters

- Can bring the dendrogram down to a certain level to get a flat clustering

```python
from scipy.cluster.hierarchy import fcluster

hier_labels = fcluster(linkage_array, 8, criterion='maxclust')
```
