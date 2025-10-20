from sklearn.datasets import load_diabetes
import pandas as pd 

def classification_dataset():
  df = load_diabetes(as_frame=True)
  df = pd.concat((df.data, df.target), axis=1)
  return df