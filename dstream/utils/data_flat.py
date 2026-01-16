import pandas as pd


def data_merge(data1, data2, on, how='inner'):
    return pd.merge(data1, data2, on=on, how=how)

def data_drop(data, columns):
    return data.drop(columns=columns)


def data_outliers(data, numerical_features):
    # Calculate the first quartile (Q1) and third quartile (Q3)
    Q1 = data[numerical_features].quantile(0.25)
    Q3 = data[numerical_features].quantile(0.75)

    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = (data[numerical_features] < lower_bound) | (data[numerical_features] > upper_bound)
    
    return outliers