from dstream.preprocess.base import IDataImputer, IDataEncoder, IDataScaler, IDataSplitter, IFeatureSelector
from dstream.preprocess.huggingface import HuggingFaceDataset
from dstream.utils.logged import setLogging
import pandas as pd
from typing import List, Optional, Tuple
from dstream.preprocess.imputer import DataImputer
from dstream.preprocess.encoder import DataEncoder
from dstream.preprocess.scaler import DataScaler
from dstream.preprocess.splitter import SimpleDataSplitter 

logger = setLogging().getLogger('Transformation')

class DataTransformer:
    def __init__(self,
                imputer: Optional[IDataImputer] = None,
                encoder: Optional[IDataEncoder] = None,
                scaler: Optional[IDataScaler] = None,
                splitter: Optional[IDataSplitter] = None,
                selector: Optional[IFeatureSelector] = None
                ):
        self.imputer = imputer
        self.encoder = encoder
        self.scaler = scaler
        self.splitter = splitter or SimpleDataSplitter()
        self.selector = selector

    @classmethod
    def from_default(cls):
        #selector=FeatureSelector(k=2, task_type='classification')

        return cls(imputer=DataImputer(), encoder=DataEncoder(categorical_cols=[]), 
                   scaler=DataScaler(), splitter=SimpleDataSplitter())
        

    def run(self, data: pd.DataFrame, features: Tuple[List[str]], target: str):
        try:
            numerical_features, categorical_features = features

            X1 = data[numerical_features]
            X2 = data[categorical_features]
            y = data[target]

            if self.imputer:
                logger.info("Performing Imputation")
                X1 = self.imputer.impute(X1)
            if self.encoder:
                logger.info("Performing Encoding")
                X2 = self.encoder.encode(X2, categorical_features)
            if self.scaler: 
                logger.info("Performing Standardization")
                X1, _ = self.scaler.scale(X1)
            if self.selector:
                logger.info("Performing Feature Selection")
                X1 = self.selector.select(X1, y)

            logger.info("Preprocessing complete.")
            return pd.concat([X1, X2], axis=1), y

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise

     
    def split_data(self, X, y):

        X_train, X_eval, y_train, y_eval = self.splitter.split(X, y)

        return X_train, X_eval, y_train, y_eval


    @staticmethod
    def to_huggingface_dataset(X_train, X_test, y_train, y_test):
        converter = HuggingFaceDataset()
        result = converter.to_huggingface_dataset(X_train, X_test, y_train, y_test)
        logger.info("Preprocessing complete with Hugging Face dataset output.")
        return result

