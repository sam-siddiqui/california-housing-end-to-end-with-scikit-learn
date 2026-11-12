"""
California Housing, End to End with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_housing
def load_housing():
    # TODO: Download housing.tgz once into tempfile.gettempdir() and read housing/housing.csv from it.
    url = "https://github.com/ageron/data/raw/main/housing.tgz"
    target_folder = tempfile.gettempdir()
    filename = url.split("/")[-1]
    full_path = os.path.join(target_folder, filename)
    os.makedirs(target_folder, exist_ok=True)
    
    if not os.path.exists(full_path):
        urllib.request.urlretrieve(url, full_path)

    data = None
    with tarfile.open(full_path, "r:gz") as tar:
        csv = tar.extractfile("housing/housing.csv")
        data = pd.read_csv(csv)

    return data
    pass

# Step 2 - income_categories
def income_categories(df):
    # TODO: pd.cut median_income with edges [0, 1.5, 3, 4.5, 6, inf] and labels 1..5; return an int Series.
    bins = [0, 1.5, 3, 4.5, 6, np.inf]
    labels = [1, 2, 3, 4, 5]

    return pd.cut(df['median_income'], bins=bins, labels=labels).astype(int)
    pass

# Step 3 - stratified_split
def stratified_split(df, test_size=0.2, random_state=42):
    # TODO: train_test_split stratified on income_categories(df); return (train_set, test_set).
    from sklearn.model_selection import train_test_split 
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=income_categories(df))
    pass

# Step 4 - explore_correlations
def explore_correlations(df):
    # TODO: Pearson correlation of every numeric column with median_house_value, sorted descending, target excluded.
    corr_matrix = df.corr(method='pearson', numeric_only=True)
    median_corr = corr_matrix['median_house_value']
    median_corr = median_corr.drop('median_house_value')
    return median_corr.sort_values(ascending=False)
    pass

# Step 5 - add_ratio_features
def add_ratio_features(df):
    # TODO: Return a copy with rooms_per_house, bedrooms_ratio and people_per_house columns added.
    new_df = df.copy(deep=True)

    ratio = lambda num, den: lambda d: d[num] / d[den]
    ratio_dict = {
        'rooms_per_house':  ratio('total_rooms', 'households'),
        'bedrooms_ratio':   ratio('total_bedrooms', 'total_rooms'),
        'people_per_house': ratio('population', 'households')
    }
    
    return new_df.assign(
        **{col: func(new_df) for col, func in ratio_dict.items()}
        # Unpacks it to a keyword arguments
    )
    pass

# Step 6 - split_features_labels
def split_features_labels(df):
    # TODO: Return (X without median_house_value, y = median_house_value Series).
    #extract_col = lambda df, col_name: df[col_name], df.copy(deep=True).drop(col_name) 
    #return extract_col(df, 'median_house_value')
    col_name = 'median_house_value'
    y = df[col_name]
    X = df.drop(columns=[col_name]).copy()
    return X, y
    pass

# Step 7 - ClusterSimilarity
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        # TODO: store the parameters
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
        pass

    def fit(self, X, y=None, sample_weight=None):
        # TODO: fit KMeans(n_clusters, n_init=10, random_state) on X with sample_weight; keep it as self.kmeans_
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters, 
            n_init=10, 
            random_state=self.random_state
        )
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        # TODO: rbf_kernel similarity of each row of X to the cluster centers
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
        pass

    def get_feature_names_out(self, names=None):
        # TODO: ["Cluster 0 similarity", ...]
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]
        pass

# Step 8 - numeric_pipeline
def numeric_pipeline():
    # TODO: make_pipeline(SimpleImputer(median), StandardScaler())
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    return make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    )
    pass

# Step 9 - categorical_pipeline
def categorical_pipeline():
    # TODO: make_pipeline(SimpleImputer(most_frequent), OneHotEncoder(handle_unknown='ignore'))
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
    from sklearn.pipeline import make_pipeline

    return make_pipeline(
        SimpleImputer(strategy="most_frequent", missing_values=None),
        OneHotEncoder(handle_unknown="ignore")
    )
    pass

# Step 10 - build_preprocessing
def build_preprocessing(n_clusters=10, gamma=1.0, random_state=42):
    # TODO: ColumnTransformer with 'log', 'geo', 'cat' transformers and remainder=numeric_pipeline().
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    
    log_cols = ["total_bedrooms", "total_rooms", "population", "households", "median_income"]
    geo_cols = ["latitude", "longitude"]
    cat_cols = ["ocean_proximity"]
    
    return ColumnTransformer(
        transformers=[
            ('log', 
                make_pipeline(
                    SimpleImputer(strategy='median'),
                    FunctionTransformer(np.log, feature_names_out="one-to-one"),
                    StandardScaler()
                ),
            log_cols),
            ('geo',
                ClusterSimilarity(n_clusters, gamma, random_state),
                geo_cols
            ),
            ('cat', categorical_pipeline(), cat_cols)
        ],
        remainder=numeric_pipeline()
    )
    pass

# Step 11 - rmse
def rmse(y_true, y_pred):
    # TODO: sqrt(mean((y_true - y_pred)^2)) as a float.
    y_t, y_p = np.array(y_true), np.array(y_pred)
    return float(np.sqrt(np.mean((y_t - y_p) ** 2)))
    pass

# Step 12 - dummy_baseline_rmse
def dummy_baseline_rmse(X, y):
    # TODO: fit DummyRegressor(strategy='mean') and return its RMSE on (X, y).
    from sklearn.dummy import DummyRegressor
    regressor = DummyRegressor(strategy="mean")
    regressor.fit(X, y)
    return rmse(y, regressor.predict(X))
    pass

# Step 13 - cross_val_rmse
def cross_val_rmse(model, X, y, cv=3):
    # TODO: cross_val_score with neg_root_mean_squared_error; return {'scores': [...], 'mean': ..., 'std': ...}.
    from sklearn.model_selection import cross_val_score
    scores = -cross_val_score( # Negate it, Because of 
        model, 
        X, 
        y=y, 
        groups=None, 
        scoring="neg_root_mean_squared_error", # this 
        cv=cv, 
        n_jobs=None
    )
    return {
        "scores": [float(score) for score in scores],
        "mean": scores.mean(),
        "std": scores.std()
    }
    pass

# Step 14 - linear_model
def linear_model(preprocessing):
    # TODO: make_pipeline(preprocessing, LinearRegression())
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LinearRegression
    return make_pipeline(preprocessing, LinearRegression())
    pass

# Step 15 - forest_model
def forest_model(preprocessing, n_estimators=50, random_state=42):
    # TODO: make_pipeline(preprocessing, RandomForestRegressor(n_estimators, random_state))
    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import RandomForestRegressor
    return make_pipeline(preprocessing, RandomForestRegressor(n_estimators=n_estimators, random_state=random_state))
    pass

# Step 16 - random_search
def random_search(pipeline, X, y, n_iter=5, cv=3, random_state=42):
    # TODO: RandomizedSearchCV over geo n_clusters 3..10 and forest max_features 2..8; fit and return it.
    from sklearn.model_selection import RandomizedSearchCV
    param_dist = {
        'columntransformer__geo__n_clusters' : range(3, 11), # Use the elbow method
        'randomforestregressor__max_features': range(2, 9)   # ~sqrt(p)
    }
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        random_state=random_state,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )
    random_search.fit(X, y)
    return random_search
    pass

# Step 17 - test_rmse
def test_rmse(model, test_set):
    # TODO: add ratio features, split, predict with the fitted model, return rmse.
    X, y = split_features_labels(add_ratio_features(test_set))
    return rmse(y, model.predict(X))
    pass

# Step 18 - bootstrap_rmse_ci
def bootstrap_rmse_ci(y_true, y_pred, n_boot=200, alpha=0.05, random_state=42):
    # TODO: bootstrap the RMSE by resampling index pairs; return (low, high) percentiles as floats.
    rmses = []
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    rng = np.random.default_rng(random_state)
    for i in range(n_boot):
        indices = rng.integers(0, len(y_true), len(y_true))
        rmses.append(rmse(y_true[indices], y_pred[indices]))
    rmses = np.array(rmses)
    return np.percentile(rmses, 100*alpha/2), np.percentile(rmses, 100*(1-alpha/2)) 
    pass

# Step 19 - feature_importances
def feature_importances(search, k=5):
    # TODO: top-k (importance, name) tuples from the best estimator, importances rounded to 3 decimals.
    return sorted(
        (
            (round(importance, 3), name) 
            for importance, name in 
            zip(
                search.best_estimator_.steps[-1][1].feature_importances_, 
                search.best_estimator_[:-1].get_feature_names_out()
            )
        ), 
        
        reverse=True, 
        key=lambda tup: tup[0]
    )[:k]
    pass

# Step 20 - worst_errors
def worst_errors(model, df, k=5):
    # TODO: DataFrame of the k largest absolute errors with columns actual, predicted, abs_error.
    col_name = 'median_house_value'
    X, y = split_features_labels(add_ratio_features(df))
    y_true = y
    y_pred = model.predict(X)

    new_df = pd.DataFrame({
        'actual': y_true,
        'predicted': y_pred,
        'abs_error': abs(y_pred - y_true)
    }, index=df.index)
    new_df = new_df.sort_values(by='abs_error', ascending=False)
    return new_df.iloc[:k]
    pass

# Step 21 - save_and_reload
def save_and_reload(model, path):
    # TODO: joblib.dump then joblib.load; return the reloaded model.
    import joblib
    joblib.dump(model, path)
    return joblib.load(path)
    pass

# Step 22 - predict_new
def predict_new(model, districts):
    # TODO: DataFrame from the dicts, add ratio features, predict, return floats rounded to the dollar.
    #data = districts
    #round_float = lambda y: [round(float(val)) * 1.0 for val in y]
    #for func in [
    #    pd.DataFrame, 
    #    add_ratio_features, 
    #    model.predict, 
    #    round_float
    #]:
    #    data = func(data)
    #return data

    df = pd.DataFrame(districts)

    df = add_ratio_features(df)

    predictions = model.predict(df)

    return [float(round(val)) for val in predictions]
    pass

