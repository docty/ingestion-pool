from sklearn.preprocessing import MinMaxScaler
import pandas as pd
 
def normalize_data(data):
    print("Normalized Data...")
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(data)
    toDF = pd.DataFrame(normalized)
    display(toDF.head())
    return normalized