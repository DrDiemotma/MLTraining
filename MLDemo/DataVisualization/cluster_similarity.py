import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

from MLDemo import DataPulling


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters: int = 10, gamma: float = 1.0, random_state=None):
        self.n_clusters: int = n_clusters
        self.gamma = gamma
        self.random_state = random_state
        self.kmeans_: KMeans | None= None

    def fit(self, x, y=None, sample_weight=None):
        x = pd.DataFrame(x)
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(x, sample_weight=sample_weight)
        return self

    def transform(self, x):
        x = pd.DataFrame(x)
        return rbf_kernel(x, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


if __name__ == "__main__":
    cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
    housing_data = DataPulling.open_tgz(DataPulling.HOUSING)
    _, housing = housing_data[0]
    print(cluster_simil.fit_transform(housing[["latitude", "longitude"]]))
