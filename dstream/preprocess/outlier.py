import pandas as pd 
from dstream.utils.logged import setLogging
import numpy as np
 
logger = setLogging()

class OutlierRemover:
    """Removes outliers using IQR method."""

    def __init__(self, factor=1.5):
        self.factor = factor

    def apply(self, data: pd.DataFrame):
        logger.info("Removing outliers using IQR method.")
        numeric_data = data.select_dtypes(include=np.number)
        Q1 = numeric_data.quantile(0.25)
        Q3 = numeric_data.quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((numeric_data < (Q1 - self.factor * IQR)) | (numeric_data > (Q3 + self.factor * IQR))).any(axis=1)
        return data.loc[mask]
