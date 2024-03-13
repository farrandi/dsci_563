# Lecture 7: Recommender Systems

## Recommender Systems Introduction

- A recommender suggests a particular product or service to users they are likely to consume.
- **Why is it important?**
  - Everything we buy or consume is influenced by this (music, shopping, movies, etc.)
  - It is the core of success for many companies (e.g. spotify, amazon, netflix, etc.)
  - Tool **to reduce the effort of users** to find what they want
- **Ethical considerations**:
  - Can lead to **filter bubbles** (e.g. political views, etc.)
  - Can lead to **privacy issues** (e.g. tracking user behavior)

### Data and Approaches to Recommender Systems

- **Data**:
  - purchase history
  - user-system interactions (e.g. clicks, likes, etc.)
  - features of the items (e.g. genre, price, etc.)
- **Approaches**:
  - **Collaborative filtering**:
    - Unsupervised learning
    - Have labels $y_{ij}$ (ratings for user $i$ and item $j$)
    - Learn latent features of users and items
  - **Content-based filtering**:
    - Supervised learning
    - Extract features of items/ users to predict ratings
  - **Hybrid methods**:
    - Combine both approaches

## Recommender Systems Structure

### Utility Matrix

- Also referred to as the **$Y$ matrix**
- Not actually used in real life because it will be very large (also sparse)
- It will store users in rows and items in columns and the values will be the ratings
- Train and validation will have same number of rows (users) and columns (items)
  - $N$ users and $M$ items
  - $Y_{ij}$ is the rating of user $i$ for item $j$

<img src="images/7_util_mat.png" width="400">

- **predict rating $\neq$ regression or classification**:
  - It is a different problem because we don't have a target variable
  - We have to predict the missing values in the utility matrix

#### Creating a Utility Matrix

```python
import pandas as pd
import numpy as np

ratings = pd.read_csv('ratings.csv')

N = len(np.unique(ratings[users]))
M = len(np.unique(ratings[items]))

user_mapper = dict(zip(np.unique(ratings['users']), list(range(N))))
item_mapper = dict(zip(np.unique(ratings['items']), list(range(M))))
user_inv_mapper = dict(zip(list(range(N)), np.unique(ratings['users'])))
item_inv_mapper = dict(zip(list(range(M)), np.unique(ratings['items'])))
```

- Map to get user/item id -> indices (utility matrix)
- Inverse map to get indices -> user/item id

```python
def create_Y_from_ratings(
    data, N, M, user_mapper, item_mapper, user_key="user_id", item_key="movie_id"):  # Function to create a dense utility matrix

    Y = np.zeros((N, M))
    Y.fill(np.nan)
    for index, val in data.iterrows():
        n = user_mapper[val[user_key]]
        m = item_mapper[val[item_key]]
        Y[n, m] = val["rating"]

    return Y

Y_mat = create_Y_from_ratings(toy_ratings, N, M, user_mapper, item_mapper)
```

### Evaluation

- No notion of "accurate" recommendations, but still need to evaluate
- Unsupervised learning but <u>split the data and evaluate </u>
  - **SPLIT TRAIN /VALID ON RATINGS, NOT UTILITY MATRIX**
  - Utility matrix of train and validation will be **the same**
  - Code shown below, not really going to use `y`

```python
from sklearn.model_selection import train_test_split

X = toy_ratings.copy()
y = toy_ratings[user_key]
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_mat = create_Y_from_ratings(X_train, N, M, user_mapper, item_mapper)
valid_mat = create_Y_from_ratings(X_valid, N, M, user_mapper, item_mapper)
```

- **RMSE**:
  - It is the most common metric
  - It compares the predicted ratings with the actual ratings

### Baseline Approaches

- **Global Average**:
  - Predict everything as the global average rating
  - It is a very simple model
- **Per-User Average**:
  - Predict everything as the average rating of the user
- **Per-Item Average**:
  - Predict everything as the average rating of the item
- **Per-User and Per-Item Average**:
  - Predict everything as the average of the user and the item
- **KNN**:

  - Calculate distance between examples usign features where neither value is missing

  ```python
  from sklearn.impute import KNNImputer

  # assume train_mat is the utility matrix
  imputer = KNNImputer(n_neighbors=2, keep_empty_features=True)
  train_mat_imp = imputer.fit_transform(train_mat)
  ```

#### Other possible approaches

1. **Clustering**:
   - Cluster the items, then recommend items from the same cluster
2. **Graphs and BFS**:
   - Create a graph of users and items
   - Use BFS to recommend items

## Collaborative Filtering

- Unsupervised learning
- **Intuition**:
  - People who agreed in the past are likely to agree again in future
  - Leverage social information for recommendations
- **PCA ?**:
  - To learn latent features of users and items
  - Run on utility matrix
  - **Problem**: missing values
    - PCA loss function $f(Z,W) = \sum_{i,j} ||W^TZ_{ij} - Y_{ij}||^2$
    - Cannot use SVD directly because have many missing values AND missing values make SVD undefined
  - **Solutions**:
    - Impute the values to do PCA
      - BUT, will introduce bias (distort the data)
      - Result will be dominated by the imputed values
    - Summing over only available values
      - Prone to overfitting
    - **Collaborative Filtering Loss Function**:
      - Only consider the available values
      - Add L2-reg to the loss function for W and Z
      - $f(Z,W) = \sum_{i,j} ||W^TZ_{ij} - Y_{ij}||^2 + \lambda_1||W||^2 + \lambda_2||Z||^2$
      - This improved the RMSE score bby 7% in the Netflix competition
      - Optimize using SGD (stoachastic gradient descent) and WALS (weighted alternating least squares)
- **Other Notes**:
  - Result can be outside the range of the ratings
  - Will have problems with cold start (new users or items)

### Z and W in Collaborative Filtering

<img src="images/7_zw.png" width="500">

- $Z$ is no longer the points in the new hyperplane and $W$ is no longer the weights
- **$Z$**:
  - Each row is a user
  - Maps users to latent feature of items
- **$W$**:
  - Each col is an item
  - Maps items to latent feature of users

### Using `surprise` library

- https://surprise.readthedocs.io/en/stable/index.html

```python
import surprise
from surprise import SVD, Dataset, Reader, accuracy

# Load the data
reader = Reader()
data = Dataset.load_from_df(ratings[['users', 'items', 'ratings']], reader)

# Train-test split
trainset, validset = train_test_split(data, test_size=0.2, random_state=42)

# PCA-like model
k=2
algo = SVD(n_factors=k, random_state=42)
algo.fit(trainset)

# Predictions
preds = algo.test(validset.build_testset())

# RMSE
accuracy.rmse(preds)
```

- Can also cross-validate

```python
from surprise.model_selection import cross_validate

cross_validate(algo, data, measures=['RMSE', "MAE"], cv=5, verbose=True)
```

## Distance Metrics

<img src="images/7_dist.png" width="300">

Source: [Google ML](https://developers.google.com/machine-learning/recommendation/overview/candidate-generation)

- **Cosine**:
  - $d(x,y) = \frac{x \cdot y}{||x|| \cdot ||y||}$
  - Collinear = 1, orthogonal = 0
  - It is the angle between the two vectors
- **Euclidean**:
  - $d(x,y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}$
  - It is the straight line distance between the two points
- **Dot Product**:
  - $d(x,y) = x \cdot y$
  - It is the projection of one vector onto the other
  - If vectors are normalized, it is the same as cosine similarity
