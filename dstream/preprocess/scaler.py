from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Any
import pandas as pd
from dstream.preprocess.base import IDataScaler
from dstream.utils.logged import setLogging
 

logger = setLogging().getLogger("Scaler")

class DataScaler(IDataScaler):
    def __init__(self, method: str = 'standard'):
        self.method = method.lower()
        self.scaler = StandardScaler() if self.method == 'standard' else MinMaxScaler()

    def scale(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
        logger.info(f"Scaling data using '{self.method}' method...")
        try:
            scaled = self.scaler.fit_transform(data)
            scaled_df = pd.DataFrame(scaled, columns=data.columns)
            logger.info("Data scaling complete.")
            return scaled_df, self.scaler
        except Exception as e:
            logger.error(f"Error during data scaling: {e}")
            raise  