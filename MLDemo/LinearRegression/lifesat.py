import pandas as pd
import matplotlib.pyplot as plt

import DataPulling
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

GDP_PER_CAPITA: str = "GDP per capita (USD)"
LIFE_SATISFACTION: str = "Life satisfaction"
X_CYPRUS: float = 37655.2

def plot_lifesat(lifesat: pd.DataFrame):
    x_data = lifesat[[GDP_PER_CAPITA]].values
    labels = lifesat[[LIFE_SATISFACTION]].values

    model_linear = LinearRegression()
    model_linear.fit(x_data, labels)
    predicted_linear = model_linear.predict([[X_CYPRUS]])[0][0]

    model_knn = KNeighborsRegressor(n_neighbors=3)
    model_knn.fit(x_data, labels)
    predicted_knn = model_knn.predict([[X_CYPRUS]])[0][0]

    lifesat.plot(kind='scatter', grid=True, x=GDP_PER_CAPITA, y=LIFE_SATISFACTION)
    plt.axis([23500, 62500, 4, 9])
    plt.title(f"Predicted value linear: {predicted_linear}, KNN: {predicted_knn}")
    plt.scatter([X_CYPRUS], [predicted_linear], c="red")
    plt.scatter([X_CYPRUS], [predicted_knn], c="green")
    plt.show()
    pass

def run():
    lifesat: pd.DataFrame | None = DataPulling.open_csv(DataPulling.LIFE_SAT)
    if lifesat is None:
        return

    plot_lifesat(lifesat)

if __name__ == "__main__":
    run()
