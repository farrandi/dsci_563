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
  - Loss function: minimize the reconstruction error (equal to getting the maximum variance)
- **Approach**
  - Find the lower dimension hyperplane that minimizes the reconstruction error
  - Model is the best-fit hyperplane

### PCA Terminology

$$X = ZW$$

$$(n \times d) = (n \times k) \cdot (k \times d)$$

Usually $k << d$

- $X$: original data, ($n \times d$)
- $Z$: lower dimension hyperplane, ($n \times k$)
- $W$: coordinates in the lower dimension, ($k \times d$)

Can **reconstruct** the original data with some error:

$$\hat{X} = ZW$$

_Note_: if $k = d$, then $Z$ is not necessarily $X$ but $\hat{X} = X$ (Frobenius norm)

- **Objective Function**:
  - Minimize $\|ZW - X\|_F^2$
    - Frobeinus norm
  - NOT the same as least squares
    - LS is vertical distance, PCA is orthogonal distance

### PCA Math

- Singular Value Decomposition (SVD)

  $$A = USV^T$$

  - $A$: original data, ($n \times d$)
  - $U$: left singular vectors, ($n \times n$)
  - $S$: diagonal matrix of singular values, ($n \times d$)
    - square root of **eigenvalues** of $A^TA$ or $AA^T$
    - corresponds to the variance of the data along the principal components (in decreasing order)
  - $V$: right singular vectors, ($d \times d$)
    - orthonormal columns (**eigenvectors**)
    - principal components (get the first $k$ columns)

- For dimensionality reduction, we can use the first $k$ columns of $U$ and the first $k$ rows of $V^T$.

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

### PCA Applications

1. Data Compression
2. Feature Extraction
3. Visualization of High-Dimensional Data
4. Dimensionality Reduction

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
