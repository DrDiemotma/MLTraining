import matplotlib.pyplot as plt
from numpy.ma.core import remainder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics import root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor

from MLDemo import DataPulling
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline, Pipeline
from MLDemo.DataVisualization import pipelines
from MLDemo.DataVisualization.cluster_similarity import ClusterSimilarity
from MLDemo.DataVisualization.pipelines import ratio_pipeline

OCEAN_PROXIMITY: str = "ocean_proximity"
MEDIAN_INCOME: str = "median_income"
MEDIAN_HOUSE_VALUE: str = "median_house_value"
TOTAL_ROOMS: str = "total_rooms"
HOUSING_MEDIAN_AGE: str = "housing_median_age"
HOUSEHOLDS: str = "households"
TOTAL_BEDROOMS: str = "total_bedrooms"
LATITUDE: str = "latitude"
LONGITUDE: str = "longitude"
POPULATION: str = "population"
INCOME_CAT: str = "housing_cat"
ROOMS_PER_HOUSE: str = "rooms_per_house"
BEDROOMS_RATIO: str = "bedrooms_ratio"
PEOPLE_PER_HOUSE: str = "people_per_house"
RANDOM_STATE: int = 42
BINS: list[float] = [0., 1.5, 3.0, 4.5, 6., np.inf]
N_SPLITS: int = 10
TEST_SIZE: float = 0.2

def print_break_line(rep: int = 20):
    print("\n" + rep * "=" + "\n")

if __name__ == "__main__":
    plt.figure()
    data_fields: list[tuple[str, pd.DataFrame]] = DataPulling.open_tgz(DataPulling.HOUSING)
    housing: pd.DataFrame = data_fields[0][1]
    housing_backup: pd.DataFrame = housing.copy()

    print(housing.head())
    print(housing.info())
    print(housing[OCEAN_PROXIMITY].value_counts())
    housing[INCOME_CAT] = pd.cut(housing[MEDIAN_INCOME], bins=BINS, labels=range(1, len(BINS)))

    print_break_line()
    print(housing.columns)


    splitter = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_set, test_set = train_test_split(housing, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    strat_train_set, strat_test_set = train_test_split(housing, test_size=TEST_SIZE, stratify=housing[INCOME_CAT], random_state=42)
    print(strat_test_set[INCOME_CAT].value_counts() / len(strat_test_set))
    print_break_line()
    test_set.drop(INCOME_CAT, axis=1, inplace=True)
    train_set.drop(INCOME_CAT, axis=1, inplace=True)

    corr_matrix: pd.DataFrame = strat_train_set.corr(numeric_only=True)
    print(corr_matrix[MEDIAN_HOUSE_VALUE].sort_values(ascending=False))

    housing: pd.DataFrame = strat_train_set.copy()

    print(f"Size of housing: {housing.shape}")
    housing[ROOMS_PER_HOUSE] = housing[TOTAL_ROOMS] / housing[HOUSEHOLDS]
    housing[BEDROOMS_RATIO] = housing[TOTAL_BEDROOMS] / housing[TOTAL_ROOMS]
    housing[PEOPLE_PER_HOUSE] = housing[POPULATION] / housing[HOUSEHOLDS]

    print_break_line()

    corr_matrix: pd.DataFrame = housing.corr(numeric_only=True)
    print(corr_matrix[MEDIAN_HOUSE_VALUE].sort_values(ascending=False))

    print_break_line()

    housing: pd.DataFrame = strat_train_set.drop([MEDIAN_HOUSE_VALUE], axis=1)
    housing_labels: pd.DataFrame = strat_train_set[MEDIAN_HOUSE_VALUE].copy()

    # median = housing[TOTAL_BEDROOMS].median()
    # housing[TOTAL_BEDROOMS].fillna(median, inplace=True)

    imputer: SimpleImputer = SimpleImputer(strategy="median")
    housing_num: pd.DataFrame = housing.select_dtypes(include=[np.number])

    imputer.fit(housing_num)
    housing_tr: pd.DataFrame = pd.DataFrame(imputer.transform(housing_num), columns=housing_num.columns,
                                            index=housing_num.index)

    print_break_line()

    isolation_forest: IsolationForest = IsolationForest(random_state=42)
    outlier_prediction = isolation_forest.fit_predict(imputer.transform(housing_num))
    print(outlier_prediction)

    print_break_line()

    housing_cat = housing[[OCEAN_PROXIMITY]]
    print(housing_cat.head(8))

    ordinal_encoder: OrdinalEncoder = OrdinalEncoder()
    housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
    one_hot_encoder: OneHotEncoder = OneHotEncoder()
    housing_cat_1hot = one_hot_encoder.fit_transform(housing_cat)

    min_max_scaler: MinMaxScaler = MinMaxScaler()
    housing_num_min_max_scaled: pd.DataFrame = min_max_scaler.fit_transform(housing_num)
    std_scaler: StandardScaler = StandardScaler()
    housing_num_std_scaled: pd.DataFrame = std_scaler.fit_transform(housing_num)

    age_simil_35 = rbf_kernel(housing[[HOUSING_MEDIAN_AGE]], [[35]], gamma=0.1)

    print_break_line()

    target_scaler = StandardScaler()
    scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

    model: LinearRegression = LinearRegression()
    # model.fit(housing[[MEDIAN_INCOME]], scaled_labels)

    # scaled_predictions = model.predict(housing[[MEDIAN_INCOME]].iloc[:5])
    # print(scaled_predictions)

    print_break_line()

    num_pipeline: Pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    housing_num_prepared: pd.DataFrame = num_pipeline.fit_transform(housing_num)

    df_housing_num_prepared: pd.DataFrame = pd.DataFrame(
        housing_num_prepared, columns=num_pipeline.get_feature_names_out(), index=housing_num.index
    )

    print(df_housing_num_prepared[:2].round(2))

    print_break_line()
    cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=RANDOM_STATE)
    default_num_pipeline: Pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    housing.drop(INCOME_CAT, axis=1, inplace=True)

    preprocessing: ColumnTransformer = ColumnTransformer([
        (BEDROOMS_RATIO, pipelines.ratio_pipeline(), [TOTAL_BEDROOMS, TOTAL_ROOMS]),
        (ROOMS_PER_HOUSE, pipelines.ratio_pipeline(), [TOTAL_ROOMS, HOUSEHOLDS]),
        (PEOPLE_PER_HOUSE, pipelines.ratio_pipeline(), [POPULATION, HOUSEHOLDS]),
        ("log", pipelines.log_pipeline(), [TOTAL_BEDROOMS, TOTAL_ROOMS, POPULATION, HOUSEHOLDS, MEDIAN_INCOME]),
        ("geo", cluster_simil, [LATITUDE, LONGITUDE]),
        ("cat", pipelines.cat_pipeline(), make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)

    housing_prepared = preprocessing.fit_transform(housing)

    print_break_line()

    lin_reg = make_pipeline(preprocessing, LinearRegression())
    lin_reg.fit(housing, housing_labels)

    housing_predictions = lin_reg.predict(housing)
    print(housing_predictions[:5].round(-2))

    lin_rmse: float = root_mean_squared_error(housing_labels, housing_predictions)
    print(f"RMSE linear regression: {lin_rmse}")

    tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=RANDOM_STATE))
    tree_reg.fit(housing, housing_labels)

    tree_predictions = tree_reg.predict(housing)
    tree_rmse: float = root_mean_squared_error(housing_labels, tree_predictions)
    print(f"RMSE tree regression: {tree_rmse}")

    print_break_line()
    print("Tree Regressor")
    tree_rmses = -cross_val_score(tree_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
    print(pd.Series(tree_rmses).describe())

    print_break_line()

    forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=RANDOM_STATE))
    forest_rmses = -cross_val_score(forest_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
    print("Random Forest Regressor")
    print(pd.Series(forest_rmses).describe())

    pass