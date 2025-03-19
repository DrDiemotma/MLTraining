from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
import pandas as pd


class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean: bool = True):  # no args or kwargs!
        self.with_mean = with_mean
        self.mean_: float = float('NaN')  # values with underscore at the end are looked for
        self.scale_: float = float('NaN')
        self.n_features_in_: int = 0

    def fit(self, x, y=None):
        """
        Fit the model to data.
        :param x: Data to transform to. In this case, it takes the mean and standard deviation out of the data frame.
        :param y: Not used here.
        :return: self.
        """
        x: pd.DataFrame = pd.DataFrame(check_array(x))
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.n_features_in_ = x.shape[1]

        return self  # very important! Always return self!

    def transform(self, x):
        """
        Transform the data to fit its predefined outcome.
        :param x: Data to transform to.
        :return: Transformed data.
        """
        check_is_fitted(self)  # this checks for tailing underscores. Not very useful since a good programmer initializes everything...
        x: pd.DataFrame = pd.DataFrame(check_array(x))
        assert self.n_features_in_ == x.shape[1]
        if self.with_mean:
            x = x - self.mean_
        return x / self.scale_
    
