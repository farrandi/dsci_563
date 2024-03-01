# Lecture 4: PCA, LSA, NMF

## LSA (Latent Semantic Analysis)

- Do not center the data and just use SVD
- Useful for sparse data (e.g. text data in a bag-of-words model)
- It is also referred to as **Latent Semantic Indexing (LSI)**
- `TruncatedSVD` in `sklearn` is used for LSA

```python
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline

lsa_pipe = make_pipeline(CountVectorizer(stop_words='english'),
                             TruncatedSVD(n_components=2))

lsa_transformed = lsa_pipe.fit_transform(df['text'])
```

## NMF (Non-Negative Matrix Factorization)

- Useful for when data is created with several independent sources
  - e.g. music with different instruments
- **Properties**:
  - Coefficients and basis vectors (components) are **non-negative**
    - Unlike in PCA you can subtract, e.g. $X_i = 14W_0 - 2W_2$
    - Since cannot cancel out => **more interpretable**
  - Components are neither orthogonal to each other nor are they sorted by the variance explained by them
  - Data is not centered
  - Will get **different results for different number of components**
    - `n_components=2` will point at extreme, `n_components=1` will point at mean
    - Unlike PCA, where first component points at the direction of maximum variance regardless of the number of components
  - Slower than PCA

## Comparison of PCA, LSA, and NMF

| Differenciating Feature | PCA                                                 | NMF                                               | LSA                                                            |
| ----------------------- | --------------------------------------------------- | ------------------------------------------------- | -------------------------------------------------------------- |
| Primary Use             | Dimensionality reduction, feature extraction        | Feature extraction, source separation             | Dimensionality reduction, semantic analysis                    |
| Data types/ constraints | Linear data, centered data                          | Non-negative data, non-centered data              | Sparse data (e.g., text data), not centered                    |
| Output components       | Orthogonal components, sorted by variance explained | Non-negative components, not orthogonal or sorted | Components from SVD, not necessarily orthogonal or sorted      |
| Interpretability        | Less interpretable due to orthogonality             | More interpretable due to non-negativity          | More interpretable, particularly in semantic analysis contexts |
