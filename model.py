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

# Step 10 - build_preprocessing (not yet solved)
# TODO: implement

# Step 11 - rmse (not yet solved)
# TODO: implement

# Step 12 - dummy_baseline_rmse (not yet solved)
# TODO: implement

# Step 13 - cross_val_rmse (not yet solved)
# TODO: implement

# Step 14 - linear_model (not yet solved)
# TODO: implement

# Step 15 - forest_model (not yet solved)
# TODO: implement

# Step 16 - random_search (not yet solved)
# TODO: implement

# Step 17 - test_rmse (not yet solved)
# TODO: implement

# Step 18 - bootstrap_rmse_ci (not yet solved)
# TODO: implement

# Step 19 - feature_importances (not yet solved)
# TODO: implement

# Step 20 - worst_errors (not yet solved)
# TODO: implement

# Step 21 - save_and_reload (not yet solved)
# TODO: implement

# Step 22 - predict_new (not yet solved)
# TODO: implement

