from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    num_pipeline = Pipeline([("impute", SimpleImputer(strategy="median")),
                             ("standardize", StandardScaler()),
                             ])

    num_pipeline2 = make_pipeline((SimpleImputer(strategy="median"), StandardScaler()))

    ## use pipeline.fit(...)