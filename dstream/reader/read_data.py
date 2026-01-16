import pandas as pd
from .base import DataReaderBase


class DataReader(DataReaderBase):

    def __init__(self, input_files:str = None):
        self.input_files=input_files
        self.data = None
        
    def load_data(self, index_col=None):
        df =  pd.read_csv(self.input_files, index_col=index_col)
        self.data = df
        return df
    
    @property
    def columns(self):
        if self.data is None:
            raise ValueError("Load the data first.")
            
        object_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()
        
        return {"object": object_cols, "numeric": numeric_cols}
    
    def set_data(self, data):
        self.data = data