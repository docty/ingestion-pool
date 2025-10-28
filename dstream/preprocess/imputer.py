from dstream.preprocess.base import IDataImputer
from dstream.utils.logged import setLogging
from sklearn.impute import SimpleImputer
import pandas as pd 

logger = setLogging().getLogger('Imputation')

class DataImputer(IDataImputer):

    def __init__(self, strategy: str = 'mean'):
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=self.strategy)

    def impute(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Missing values using '{self.strategy}' strategy...")
        try:
            filled = self.imputer.fit_transform(data)
            filled_df = pd.DataFrame(filled, columns=data.columns)
            logger.info("Imputation complete.")
            return filled_df
        except Exception as e:
            logger.error(f"Error during data imputation: {e}")
            raise
