from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
 
def normalize_data(data, method='std'): 
    if method=='minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
        
    normalized = scaler.fit_transform(data)
    toDF = pd.DataFrame(normalized, columns=data.columns)
     
    return toDF