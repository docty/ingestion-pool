from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple, Any

class IDataSplitter(ABC):
    @abstractmethod
    def split(self, X, y) -> Tuple:
        pass


class IDataScaler(ABC):
    @abstractmethod
    def scale(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
        pass


class IDataImputer(ABC):
    @abstractmethod
    def impute(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class IDataEncoder(ABC):
    @abstractmethod
    def encode(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class IFeatureSelector(ABC):
    @abstractmethod
    def select(self, X, y) -> pd.DataFrame:
        pass