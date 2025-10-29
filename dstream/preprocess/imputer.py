from dstream.preprocess.base import IDataImputer
from dstream.utils.logged import setLogging
from sklearn.impute import SimpleImputer
import pandas as pd 

logger = setLogging().getLogger('Imputation')

class DataImputer(IDataImputer):

    def __init__(self, strategy: str = 'mean', fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.imputer = SimpleImputer(strategy=self.strategy, fill_value=self.fill_value)

    def impute(self, data: pd.DataFrame, column=None) -> pd.DataFrame:
        logger.info(f"Missing values using '{self.strategy}' strategy...")
         
        try:
            if column:
                filled = self.imputer.fit_transform(data[column])
                data[column] = filled
                logger.info(f"Imputation of {column} column completed.")
                return data
            
            filled = self.imputer.fit_transform(data)
            filled_df = pd.DataFrame(filled, columns=data.columns)
            logger.info("Imputation completed.")
            return filled_df
        
        except Exception as e:
            logger.error(f"Error during data imputation: {e}")
            raise
