import json
import os
import unittest

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_dataset_created(self):
        self.assertTrue(
            os.path.exists("house_prices_practice.csv")
        )

    def test_model_created(self):
        self.assertTrue(
            os.path.exists("house_price_model.pkl")
        )

    def test_metrics_created(self):
        self.assertTrue(
            os.path.exists("metrics.json")
        )

    def test_mae_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        mae = metrics["mae"]

        self.assertGreaterEqual(
            mae,
            0.0
        )

    def test_rmse_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        rmse = metrics["rmse"]

        self.assertGreaterEqual(
            rmse,
            0.0
        )

    def test_r2_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        r2 = metrics["r2_score"]

        self.assertGreaterEqual(
            r2,
            -1.0
        )

        self.assertLessEqual(
            r2,
            1.0
        )

    def test_model_prediction(self):
        model = joblib.load(
            "house_price_model.pkl"
        )

        sample = pd.DataFrame([{
            "OverallQual": 7,
            "GrLivArea": 1800,
            "GarageCars": 2,
            "TotalBsmtSF": 1000,
            "YearBuilt": 2005,
            "FullBath": 2,
            "BedroomAbvGr": 3,
            "LotArea": 8000
        }])

        prediction = model.predict(sample)[0]

        self.assertGreater(
            prediction,
            0
        )


if __name__ == "__main__":
    unittest.main()
