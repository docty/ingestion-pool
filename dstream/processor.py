from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

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
     
    return toDF




def tsne():
    model = TSNE()
    
    transformed_model = model.fit_transform(df)
    
    xs = transformed_model[:,0]
    ys =transformed_model[:,1]
    return xs, ys