from sklearn.datasets import (
    load_diabetes,
    load_iris,
    make_regression as make_reg,
    make_classification as make_class
)
import pandas as pd


class Dataset:
    @staticmethod
    def load_regression():
        return load_diabetes(as_frame=True).frame

    @staticmethod
    def load_classification():
        return load_iris(as_frame=True).frame

    @staticmethod
    def make_regression(n_samples=100, n_features=2, noise=0.0, random_state=None):
        features, target = make_reg(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
            random_state=random_state
        )
        data = pd.DataFrame(features, columns=[f"x{i+1}" for i in range(n_features)])
        data["target"] = target
        return data

    @staticmethod
    def make_classification(
        n_samples=100,
        n_features=3,
        n_classes=2,
        n_informative=None,
        n_redundant=0,
        random_state=None
    ):
        features, target = make_class(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_redundant=n_redundant,
            n_informative=n_informative or n_features,
            random_state=random_state
        )
        data = pd.DataFrame(features, columns=[f"x{i+1}" for i in range(n_features)])
        data["target"] = target
        return data
