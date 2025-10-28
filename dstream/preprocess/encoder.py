from dstream.preprocess.base import IDataEncoder
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
from typing import List, Optional, Tuple, Any
from abc import ABC, abstractmethod
from dstream.utils.logged import setLogging
 
logger = setLogging().getLogger('Encoding')

class DataEncoder(IDataEncoder):
    """Encodes categorical variables using OneHotEncoder."""
    
    def __init__(self, categorical_cols: Optional[List[str]] = None):
        self.categorical_cols = categorical_cols or []
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    def encode(self, data: pd.DataFrame, categorical_features=None) -> pd.DataFrame:
        categorical_cols = self.categorical_cols or categorical_features
        if not categorical_cols:
            logger.info("No categorical columns to encode.")
            return data

        logger.info(f"Encoding categorical columns: {categorical_cols}")
        try:
            encoded = self.encoder.fit_transform(data[categorical_cols])
            encoded_df = pd.DataFrame(
                encoded,
                columns=self.encoder.get_feature_names_out(categorical_cols),
                index=data.index,
            )
            result = pd.concat([data.drop(columns=categorical_cols), encoded_df], axis=1)
            logger.info("Encoding complete.")
            return result
        except Exception as e:
            logger.error(f"Error during encoding: {e}")
            raise


class Encoder(ABC):
    """Abstract encoder class for categorical features."""
    
    @abstractmethod
    def fit_transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
        pass


class OneHotEncoderWrapper(Encoder):
    """Encodes categorical columns using one-hot encoding."""
    
    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def fit_transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
        logger.info("Applying OneHotEncoder to categorical data.")
        encoded = self.encoder.fit_transform(data)
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(data.columns))
        return encoded_df, self.encoder


class LabelEncoderWrapper(Encoder):
    """Encodes a single categorical column using label encoding."""
    
    def __init__(self):
        self.encoder = LabelEncoder()

    def fit_transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
        logger.info("Applying LabelEncoder to categorical data.")
        encoded = data.apply(self.encoder.fit_transform)
        return encoded, self.encoder


class CategoricalEncoder:
    """Encodes categorical variables using OneHotEncoder."""

    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    def apply(self, data: pd.DataFrame):
        logger.info("Encoding categorical features.")
        cat_cols = data.select_dtypes(include=['object', 'category']).columns
        if not len(cat_cols):
            return data

        logger.info(f"Encoding categorical features: {list(cat_cols)}")
        encoded = self.encoder.fit_transform(data[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(cat_cols))
        
        data = data.drop(columns=cat_cols).reset_index(drop=True)
        encoded_df = encoded_df.reset_index(drop=True)
        return pd.concat([data, encoded_df], axis=1)


