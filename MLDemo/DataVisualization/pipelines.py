import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder

from MLDemo.CustomRegressor.knn_transformer import KnnTransformer


def column_ratio(x: pd.DataFrame, index0: int = 0, index1: int = 1) -> pd.DataFrame:
    """
    Returns the ratio of two columns in the data frame.
    :param x: The data frame to use.
    :param index0: First column.
    :param index1: Second column.
    :return: Ratio.
    """
    return x[:, [index0]] / x[:, [index1]]

def ratio_pipeline(imputer_strategy: str = "median", index0: int = 0, index1: int = 1) -> Pipeline:
    """
    Create a new ratio pipline. Simple imputer -> Column ratio -> Standard scaler.
    :param imputer_strategy: Strategy for the imputer. Default: 'median'.
    :param index0: First column index for the ratio. Default: 0.
    :param index1: second column index for the ratio. Default: 1.
    :return: The newly created pipeline.
    """
    return make_pipeline(
        SimpleImputer(strategy=imputer_strategy),
        FunctionTransformer(lambda x: column_ratio(x, index0, index1), feature_names_out=lambda f1, f2: "ratio"),
        StandardScaler()
    )

def log_pipeline(imputer_strategy: str = "median") -> Pipeline:
    return make_pipeline(
        SimpleImputer(strategy=imputer_strategy),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler()
    )

def cat_pipeline(imputer_strategy: str = "most_frequent", handle_unknown: str = "ignore"):
    """
    Categorical pipeline. Transforms to One hots.
    :param imputer_strategy: Imputes Nones with this strategy. Default is 'most_frequent'.
    :param handle_unknown: How to handle unknown categories. Default is 'ignore'.
    :return: A pipeline which encodes categorical features.
    """
    return make_pipeline(
        SimpleImputer(strategy=imputer_strategy),
        OneHotEncoder(handle_unknown=handle_unknown)
    )

def knn_pipeline(imputer_strategy: str = "median"):
    return make_pipeline(
        SimpleImputer(strategy=imputer_strategy),
        KnnTransformer()
    )

def standardize_pipeline(imputer_strategy: str = "median"):
    return make_pipeline(
        SimpleImputer(strategy=imputer_strategy),
        StandardScaler()
    )


if __name__ == "__main__":
    num_pipeline = Pipeline([("impute", SimpleImputer(strategy="median")),
                             ("standardize", StandardScaler()),
                             ])

    ## Create a pipeline
    num_pipeline2 = make_pipeline((SimpleImputer(strategy="median"), StandardScaler()))

    ## use pipeline.fit(...)