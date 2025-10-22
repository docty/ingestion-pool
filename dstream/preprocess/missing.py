import pandas as pd 
from dstream.preprocess.utils import setLogging
import numpy as np
 
logger = setLogging()

class MissingValueHandler:
    """Handles missing values in datasets."""

    def __init__(self, method="mean"):
        self.method = method

    def _fill_with_mean(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        logger.info(f"Filling missing values in column '{column}' with mean.")
        data[column] = data[column].fillna(data[column].mean())
        return data

    def _fill_with_median(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        logger.info(f"Filling missing values in column '{column}' with mean.")
        data[column] = data[column].fillna(data[column].median())
        return data

    def _fill_with_mode(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        logger.info(f"Filling missing values in column '{column}' with mean.")
        data[column] = data[column].fillna(data[column].mode()[0])
        return data


    def apply(self, data: pd.DataFrame):
        logger.info(f"Filling missing values using method: {self.method}")
        for column in data.columns:
            if data[column].isnull().any():
                if self.method == 'mean' and np.issubdtype(data[column].dtype, np.number):
                    data[column] = self._fill_with_mean(data, column)
                elif self.method == 'median' and np.issubdtype(data[column].dtype, np.number):
                    data[column] = self._fill_with_median(data, column)
                else:
                    data[column] = self._fill_with_mode(data, column)
        return data
