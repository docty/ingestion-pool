from dstream.preprocess.base import IDataImputer, IDataEncoder, IDataScaler, IDataSplitter, IFeatureSelector
from dstream.preprocess.huggingface import HuggingFaceDataset
from dstream.preprocess.utils import setLogging
import pandas as pd
from typing import List, Optional
from dstream.preprocess.imputer import DataImputer
from dstream.preprocess.encoder import DataEncoder
from dstream.preprocess.scaler import DataScaler
from dstream.preprocess.splitter import SimpleDataSplitter 

logger = setLogging()

class DataTransformer:
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

    @classmethod
    def from_default(cls):
        #selector=FeatureSelector(k=2, task_type='classification')

        return cls(imputer=DataImputer(), encoder=DataEncoder(categorical_cols=[]), 
                   scaler=DataScaler(), splitter=SimpleDataSplitter())
        

    def run(self, data: pd.DataFrame, features: List[str], target: str):
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

            logger.info("Preprocessing complete with Pandas dataset output.")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise

    @staticmethod
    def to_huggingface_dataset(X_train, X_test, y_train, y_test):
        converter = HuggingFaceDataset()
        result = converter.to_huggingface_dataset(X_train, X_test, y_train, y_test)
        logger.info("Preprocessing complete with Hugging Face dataset output.")
        return result

