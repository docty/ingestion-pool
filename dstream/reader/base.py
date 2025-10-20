from abc import ABC, abstractmethod
import pandas as pd 

class DataReaderBase(ABC):
    @abstractmethod
    def load_data(self, index_col=None) -> pd.DataFrame:
        pass
