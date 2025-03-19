import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import BaseEstimator, TransformerMixin


class KnnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._knn_regressor: KNeighborsRegressor | None = None
        pass

    def fit(self, x, y=None):
        self._knn_regressor = KNeighborsRegressor()
        self._knn_regressor.fit(x)
        return self

    def transform(self, x):
        x = pd.DataFrame(x)
        if self._knn_regressor is None:
            return

        return self._knn_regressor.predict(x)

    def get_feature_names_out(self, names=None):
        return "Custom KNN transformer"
