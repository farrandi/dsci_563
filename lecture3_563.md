# Lecture 3: PCA

## Principal Component Analysis (PCA)

### Dimensionality Reduction Motivation

- **Curse of Dimensionality**
  - As the number of dimensions increases, the volume of the space increases so fast that the available data become sparse
- **Data Visualization**
  - It is difficult to visualize data in high dimensions
- **Computational Efficiency**
  - Many algorithms are computationally infeasible in high dimensions

### PCA Overview

- **Goal**
  - Find a low-dimensional representation of the data that captures as much of the variance as possible
- **Approach**
  - Find the lower dimension hyperplane that minimizes the reconstruction error
  - Model is the best-fit hyperplane

### PCA Terminology

$$X = ZW$$

$$(n \times d) = (n \times k) \cdot (k \times d)$$

Usually $k << d$

- $X$: original data, ($n \times d$)
- $Z$: coordinates in the lower dimension, ($n \times k$)
- $W$: lower dimension hyperplane, ($k \times d$)
  - $W$ is the principal components
  - Rows of $W$ are orthogonal to each other

Can **reconstruct** the original data with some error:

$$\hat{X} = ZW$$

_Note_: if $k = d$, then $Z$ is not necessarily $X$ but $\hat{X} = X$ (Frobenius norm)

- **Objective/ Loss Function**:
  - Minimize reconstruction error $\|ZW - X\|_F^2$
    - Frobeinus norm $||A||_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n a_{ij}^2}$
  - NOT the same as least squares
    - LS is vertical distance, PCA is orthogonal distance

### PCA Math

- Singular Value Decomposition (SVD)

  $$A = USV^T$$

  - $A$: original data, ($n \times d$)
  - $U$: **left singular vectors**, ($n \times n$)
    - orthonormal columns $U_i^TU_j = 0$ for all $i \neq j$
  - $S$: **diagonal matrix** of singular values, ($n \times d$)
    - square root of **eigenvalues** of $A^TA$ or $AA^T$
    - corresponds to the variance of the data along the principal components (in decreasing order)
  - $V$: **right singular vectors**, ($d \times d$)
    - orthonormal columns (**eigenvectors**)
    - principal components (get the first $k$ columns)

- For dimensionality reduction, we can use the first $k$ columns of $U$ and the first $k$ rows of $V^T$ (equal to first $k$ columns of $V$)

- **PCA Algorithm**

  1. Center the data (subtract the mean of each column)
  2. Compute the SVD of the centered data to get the principal components $W$.
     - $W$ is the first $k$ columns of $V$
  3. Variance of each PC is the square of the singular value $s_i^2$
  4. Drop the PCs with the smallest variance

- **Uniquess of PCA**

  - PCA are not unique, similar to eigenvectors
  - Can add constraints to make it closer to unique:
    - Normalization: $||w_i|| = 1$
    - Orthogonality: $w_i^Tw_j = 0$ for all $i \neq j$
    - Sequential PCA: $w_1^Tw_2 = 0$, $w_2^Tw_3 = 0$, etc.
  - The principal components are unique up to sign

### Choosing Number of Components

- No definitive rule
- Can look at:
  - Explained variance ratio
  - Reconstructions plot

```python
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(samples)

# Plot the explained variances
plt.plot(np.cumsum(pca.explained_variance_ratio_))
```

### PCA and Multicollinearity

- PCA can be used to remove multicollinearity
- **Concept**: the principal components are orthogonal to each other so they should not be correlated

### PCA Applications

1. **Data Compression**
   - Can use the first $k$ principal components to represent the data
2. **Feature Extraction**
   - Can use the first $k$ principal components as features
3. **Visualization of High-Dimensional Data**
   - Can visualize the data in 2D or 3D by using the first 2 or 3 principal components
4. **Dimensionality Reduction**
5. **Anomaly Detection**
   - Can use the reconstruction error to detect anomalies/ outliers (if the error is too large)
   - Outliers = high reconstruction error

### PCA in Python

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

pca = PCA(n_components=2)
pipeline = make_pipeline(StandardScaler(), pca)

# Fit the pipeline to 'samples'
Z = pipeline.fit_transform(samples)
X_hat = pipeline.inverse_transform(Z)

# Get the principal components
print(pca.components_)
```

### K-means and PCA

- PCA is a generalization of K-means
- K-means is a special case of PCA where the principal components are the cluster centers
- K-means each example is expressed with only one component (one-hot encoding) but in PCA it is a linear combination of all components
