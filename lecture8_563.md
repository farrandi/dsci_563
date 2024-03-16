# Lecture 8: Content Based Filtering

## Content Based Filtering

- Supervised learning
- Does not make use of _social network / information_
- Solves the cold start problem (can recommend items to new users/items)
- Assumes that we have **features of items and/or users** to predict ratings
- Create a user profile for each user
  - Treat rating prediction as a regression problem
  - Have a regression model for each user

### Steps in Python (With a movie recommendation example)

0. Load `ratings_df` (contains `user_id`, `movie_id`, and `rating`)
   - Also make the user and movie mappers.

```python
toy_ratings = pd.read_csv("data/toy_ratings.csv")

N = len(np.unique(toy_ratings["user_id"]))
M = len(np.unique(toy_ratings["movie_id"]))

user_key = "user_id" # Name of user
item_key = "movie_id" # Name of movie

# Turns the name into a number (id)
user_mapper = dict(zip(np.unique(toy_ratings[user_key]), list(range(N))))
item_mapper = dict(zip(np.unique(toy_ratings[item_key]), list(range(M))))

# Turns the number (id) back into a name
user_inverse_mapper = dict(zip(list(range(N)), np.unique(toy_ratings[user_key])))
item_inverse_mapper = dict(zip(list(range(M)), np.unique(toy_ratings[item_key])))
```

1. **Load movie features**. This is a matrix of shape `(n_movies, n_features)`
   - Index of movie features is movie id/name
   - Features can be genre, director, actors, etc.

```python
import pandas as pd

movie_feats_df = pd.read_csv("data/toy_movie_feats.csv", index_col=0)
Z = movie_feats_df.to_numpy()
```

<img src="images/8_1.png" width="400">

2. **Build a user profile**. For each user, we will get the ratings and the corresponding movie features.

   - Results in a dictionary (key: user, value: numpy array of size `(n_ratings, n_genres)`)
     - `n_ratings`: number of movies rated by the user

```python
from collections import defaultdict

def get_X_y_per_user(ratings, d=item_feats.shape[1]):
    """
    Returns X and y for each user.

    Parameters:
    ----------
    ratings : pandas.DataFrame
         ratings data as a dataframe

    d : int
        number of item features

    Return:
    ----------
        dictionaries containing X and y for all users
    """
    lr_y = defaultdict(list)
    lr_X = defaultdict(list)

    for index, val in ratings.iterrows():
        n = user_mapper[val[user_key]]
        m = item_mapper[val[item_key]]
        lr_X[n].append(item_feats[m])
        lr_y[n].append(val["rating"])

    for n in lr_X:
        lr_X[n] = np.array(lr_X[n])
        lr_y[n] = np.array(lr_y[n])

    return lr_X, lr_y

def get_user_profile(user_name):
  """
  Get the user profile based on the user name

  e.g. get_user_profile("user1")
  """
    X = X_train_usr[user_mapper[user_name]]
    y = y_train_usr[user_mapper[user_name]]
    items = rated_items[user_mapper[user_name]]
    movie_names = [item_inverse_mapper[item] for item in items]
    print("Profile for user: ", user_name)
    profile_df = pd.DataFrame(X, columns=movie_feats_df.columns, index=movie_names)
    profile_df["ratings"] = y
    return profile_df
```

```python
# Using the helper functions
Xt,yt = get_X_y_per_user(X_train)
Xv,yv = get_X_y_per_user(X_valid)

# Check the user profile
get_user_profile("Nando")
```

3. **Supervised learning**. Train a regression model for each user.

   - We will use `Ridge` regression model for this example.

```python
from sklearn.linear_model import Ridge

models = dict()
# Make utility matrix
pred_lin_reg = np.zeros((N, M))

for n in range(N):
  models[n] = Ridge()
  models[n].fit(Xt[n], yt[n])
  pred_lin_reg[n] = models[n].predict(item_feats)
```

## Collaborative vs Content Based Filtering

| Collaborative Filtering                                                                  | Content Based Filtering                                                                |
| ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| $\hat{y}_{ij} = w_j^T z_i$                                                               | $\hat{y}_{ij} = w_i^T x_{ij}$                                                          |
| $w_j^T$: "hidden" embedding for feature $j$ </br> $z_i$: "hidden" embedding for user $i$ | $w_i$: feature vector for user $i$ </br> $x_{ij}$: feature $j$ for user $i$            |
| (+) Makes use of social network / information                                            | (-) Does not make use of social network / information                                  |
| (-) Cold start problem                                                                   | (+) Solves the cold start problem                                                      |
| (+) No feature engineering required                                                      | (-) Requires feature engineering                                                       |
| (-) Hard to interpret                                                                    | (+) Easy to interpret                                                                  |
| (-) Cannot capture unique user preferences                                               | (+) Can capture unique user preferences (since each model is unique)                   |
| (+) More diverse recommendations                                                         | (-) Less diverse recommendations (hardly recommend an item outside the user’s profile) |

## Beyond Error Rate in Recommender Systems

- Best RMSE $\neq$ Best Recommender
- Need to consider simplicity, interpretatibility, code maintainability, etc.
  - The Netflix Prize: The winning solution was never implemented
- Other things to consider:
  - Diversity: If someone buys a tennis racket, they might not want to buy another tennis racket
  - Freshness: New items (new items need to be recommended for it to be successful)
  - Trust: Explain your recommendation to the user
  - Persistence: If the same recommendation is made over and over again, it might not be a good recommendation
  - Social influence: If a friend buys something, you might want to buy it too
- Also need to consider ethical implications
  - Filter bubbles
  - Recommending harmful content
