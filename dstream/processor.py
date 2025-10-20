from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from datasets import DatasetDict, Dataset 

import pandas as pd

def data_Xy(data, feature, target, split=False):
    X = data[feature]
    y = data[target]
    
    if split:
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        return X_train, X_test, y_train, y_test
        
    return X, y


 

def normalize_data(data, method='std'): 
    if method=='minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
        
    normalized = scaler.fit_transform(data)
    toDF = pd.DataFrame(normalized, columns=data.columns)
     
    return toDF, scaler




def tsne(data):
    model = TSNE()
    
    transformed_model = model.fit_transform(data)
    
    xs = transformed_model[:,0]
    ys =transformed_model[:,1]
    return xs, ys



 

def convert_input_label(example):
    feature_columns = [k for k in example.keys() if k != "target"]
   
    return {"input": [example[f] for f in feature_columns], "label": example["target"]}


def to_huggingface_dataset(X_train, X_test, y_train, y_test):

  train_df = pd.concat([X_train, y_train], axis=1) 
  test_df = pd.concat([X_test, y_test], axis=1) 

  dataset_dict = DatasetDict({
      "train": Dataset.from_pandas(train_df,  preserve_index=False),
      "test": Dataset.from_pandas(test_df, preserve_index=False)
  })

  new_data = dataset_dict.map(convert_input_label, remove_columns=dataset_dict['train'].features)

  return new_data


def fillna(data, column):
    results = data[column].fillna(data[column].mean())
    data[column] = results
    return data