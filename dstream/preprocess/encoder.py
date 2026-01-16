from typing import List, Optional, Tuple, Any
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer
import pandas as pd
from abc import ABC, abstractmethod
from dstream.preprocess.base import IDataEncoder
from dstream.utils.logged import setLogging

logger = setLogging().getLogger("Encoder")


class Encoder(ABC):
    """Abstract base class for categorical encoders."""

    @abstractmethod
    def fit_transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
        """Fit and transform categorical data."""
        pass


class OneHotEncoderWrapper(Encoder):
    """Wrapper for scikit-learn's OneHotEncoder."""

    def __init__(self, handle_unknown: str = 'ignore', sparse: bool = False):
        self.encoder = OneHotEncoder(handle_unknown=handle_unknown, sparse_output=sparse)

    def fit_transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
        logger.info("Applying OneHotEncoder to categorical data.")
        encoded = self.encoder.fit_transform(data)
        encoded_df = pd.DataFrame(
            encoded,
            columns=self.encoder.get_feature_names_out(data.columns),
            index=data.index
        )
        return encoded_df, self.encoder


class LabelEncoderWrapper(Encoder):
    """Wrapper for scikit-learn's LabelEncoder applied to each column."""

    def __init__(self):
        self.encoders = {}

    def fit_transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
        logger.info("Applying LabelEncoder to categorical data.")
        encoded_df = data.copy()

        for col in data.columns:
            le = LabelEncoder()
            encoded_df[col] = le.fit_transform(data[col].astype(str))
            self.encoders[col] = le

        return encoded_df, self.encoders


class LabelBinarizerWrapper(Encoder):
    """Wrapper for scikit-learn's LabelBinarizer for single-column binary encoding."""

    def __init__(self):
        self.encoder = LabelBinarizer()

    def fit_transform(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
        if data.shape[1] != 1:
            raise ValueError("LabelBinarizerWrapper expects exactly one column.")

        col = data.columns[0]
        logger.info(f"Applying LabelBinarizer to column: {col}")
        encoded = self.encoder.fit_transform(data[col].astype(str))

         
        if encoded.ndim == 1:
            encoded_df = pd.DataFrame(encoded, columns=[f"{col}_bin"], index=data.index)
        else:
            encoded_df = pd.DataFrame(encoded, columns=self.encoder.classes_, index=data.index)

        return encoded_df, self.encoder


class DataEncoder(IDataEncoder):
 
    def __init__(
        self,
        method: str = "onehot",
        categorical_cols: Optional[List[str]] = None,
        custom_encoder = None
    ):
        self.method = method.lower()
        self.categorical_cols = categorical_cols
        self.encoder: Encoder = custom_encoder or self._select_encoder()

    def _select_encoder(self) -> Encoder:
        if self.method == "onehot":
            return OneHotEncoderWrapper()
        elif self.method == "label":
            return LabelEncoderWrapper()
        elif self.method == "binarizer":
            return LabelBinarizerWrapper()
        else:
            raise ValueError(f"Unknown encoding method '{self.method}'")

    def encode(self, data: pd.DataFrame) -> pd.DataFrame:
        categorical_cols = self.categorical_cols or data.select_dtypes(include=["object", "category"]).columns.tolist()
        if not categorical_cols:
            logger.info("No categorical columns to encode.")
            return data

        logger.info(f"Encoding categorical columns: {categorical_cols}")
        try:
            if isinstance(self.encoder, LabelBinarizerWrapper):
                if len(categorical_cols) != 1:
                    raise ValueError("LabelBinarizer requires exactly one categorical column.")
                encoded_df, _ = self.encoder.fit_transform(data[categorical_cols])
                non_cat = data.drop(columns=categorical_cols)
                return pd.concat([non_cat.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

           
            encoded_df = self.encoder.fit_transform(data[categorical_cols].to_numpy())
             
            result = pd.DataFrame(
                encoded_df,
                #columns=self.encoder.get_feature_names_out(data.columns),
                index=data.index
            )
            
            logger.info("Encoding complete.")
            return result

        except Exception as e:
            logger.error(f"Error during encoding: {e}")
            raise
