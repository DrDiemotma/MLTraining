from matplotlib.pyplot import figure
from scipy.stats import alpha

import DataPulling
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer

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
RANDOM_STATE: int = 42
BINS: list[float] = [0., 1.5, 3.0, 4.5, 6., np.inf]
N_SPLITS: int = 10
TEST_SIZE: float = 0.2


def process_housing_data(housing: pd.DataFrame):
    print(housing.info())
    print(housing.head())
    print(housing[OCEAN_PROXIMITY].value_counts())
    print(housing.describe())
    ## Group the median income to find categories for stratification, so each bin is equally in each test and
    ## training set
    housing[INCOME_CAT] = pd.cut(housing[MEDIAN_INCOME], bins=BINS, labels=list(range(1, len(BINS))))

    ## Example of stratified test sets: first just once (commented out), second create a list of non-exclusive tests.
    ## Use the latter for bootstrapping.

    # train_set, test_set = train_test_split(housing, test_size=0.2, random_state=RANDOM_STATE, stratify=[INCOME_CAT])
    splitter = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    strat_splits:list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for train_index, test_index in splitter.split(housing, housing[INCOME_CAT]):
        strat_train_set_n = housing.iloc[train_index]
        strat_test_set_n = housing.iloc[test_index]
        strat_splits.append((strat_train_set_n, strat_test_set_n))

    strat_train_set, strat_test_set = strat_splits[0]
    print(strat_test_set[INCOME_CAT].value_counts() / len(strat_test_set))

    #
    # housing[INCOME_CAT].value_counts().sort_index().plot.bar(rot=0, grid=True)
    # plt.xlabel("Income category")
    # plt.ylabel("Number of districts")
    # plt.show()

    # for set_ in (strat_train_set, strat_test_set):
    #     set_.drop(INCOME_CAT, axis=1, inplace=True)

    # visualization(strat_train_set.copy().drop(INCOME_CAT, axis=1))
    data_cleaning(strat_train_set.copy().drop(INCOME_CAT, axis=1))


def visualization(housing: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(20, 14))
    housing.plot(kind="scatter", x=LONGITUDE, y=LATITUDE, grid=True, ax=ax, s=housing[POPULATION] / 100,
                 label=POPULATION, c=MEDIAN_HOUSE_VALUE, cmap="jet", colorbar=True, legend=True, sharex=False)


    corr_matrix: pd.DataFrame = housing.loc[:,
                                [
                                    MEDIAN_HOUSE_VALUE,
                                    MEDIAN_INCOME,
                                    TOTAL_ROOMS,
                                    HOUSING_MEDIAN_AGE,
                                    HOUSEHOLDS,
                                    TOTAL_BEDROOMS,
                                    POPULATION,
                                    LONGITUDE,
                                    LATITUDE
                                ]].corr()
    print(corr_matrix[MEDIAN_HOUSE_VALUE].sort_values(ascending=False))
    attributes = [MEDIAN_HOUSE_VALUE, MEDIAN_INCOME, TOTAL_ROOMS, HOUSING_MEDIAN_AGE]
    fig, ax = plt.subplots(figsize=(20, 14))
    scatter_matrix(housing[attributes], ax=ax)
    plt.show()

def data_cleaning(housing: pd.DataFrame, imputer: SimpleImputer | None = None):
    # housing.dropna(subset=[TOTAL_BEDROOMS], inplace=True)  # drop NAs

    # median = housing[TOTAL_BEDROOMS].median()
    # housing[TOTAL_BEDROOMS].fillna(median, inplace=True)  # fill NAs with the median value (in this case)

    if imputer is None:
        imputer = SimpleImputer(strategy="median")

    housing_num = housing.select_dtypes(include=[np.number])
    imputer.fit(housing_num)
    imputer.transform(housing_num)

    pass

def run():
    housing_data = DataPulling.open_tgz(DataPulling.HOUSING)
    for _, table in housing_data:
        process_housing_data(table)

if __name__ == '__main__':
    run()
