import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_model():

    print("Loading house price dataset...")

    data = pd.read_csv("house_prices_practice.csv")

    print("Dataset loaded successfully.")
    print("Number of records:", len(data))
    print("Number of columns:", len(data.columns))

    features = [
        "OverallQual",
        "GrLivArea",
        "GarageCars",
        "TotalBsmtSF",
        "YearBuilt",
        "FullBath",
        "BedroomAbvGr",
        "LotArea"
    ]

    target = "SalePrice"

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("Training records:", len(X_train))
    print("Testing records :", len(X_test))

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ])

    print("Training house price model...")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\nModel Evaluation")
    print("----------------")

    print(
        "MAE:",
        round(mae, 2)
    )

    print(
        "RMSE:",
        round(rmse, 2)
    )

    print(
        "R2 Score:",
        round(r2, 4)
    )

    joblib.dump(
        model,
        "house_price_model.pkl"
    )

    print(
        "\nModel saved as house_price_model.pkl"
    )

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2_score": float(r2),
        "training_records": len(X_train),
        "testing_records": len(X_test)
    }

    with open(
        "metrics.json",
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print(
        "Metrics saved as metrics.json"
    )


if __name__ == "__main__":

    train_model()
