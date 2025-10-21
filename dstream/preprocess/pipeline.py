#from dstream.preprocess.base import IDataImputer, IDataEncoder, IDataScaler, IDataSplitter, IFeatureSelector
#from dstream.preprocess.converter import HuggingFaceDatasetConverter
#from dstream.preprocess.utils import setLogging
import pandas as pd 
from typing import List, Optional

 
logger = setLogging()

class DataPreprocessor:
    def __init__(self,
                 imputer: Optional[IDataImputer]=None,
                 encoder: Optional[IDataEncoder]=None,
                 scaler: Optional[IDataScaler]=None,
                 splitter: Optional[IDataSplitter]=None,
                 selector: Optional[IFeatureSelector] = None):
        self.imputer = imputer
        self.encoder = encoder
        self.scaler = scaler
        self.splitter = splitter
        self.selector = selector

    def process(self, data: pd.DataFrame, features: List[str], target: str, to_hf=False):
        try:
            logger.info("Starting preprocessing pipeline...")
            X = data[features]
            y = data[target]

            X = self.imputer.impute(X)
            X = self.encoder.encode(X)
            X, _ = self.scaler.scale(X)

            if self.selector:
                X = self.selector.select(X, y)

            X_train, X_test, y_train, y_test = self.splitter.split(X, y)

            if to_hf:
                converter = HuggingFaceDatasetConverter()
                result = converter.to_huggingface_dataset(X_train, X_test, y_train, y_test)
                logger.info("Preprocessing complete with Hugging Face dataset output.")
                return result

            logger.info("Preprocessing complete with Pandas dataset output.")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise
