from IPython.display import display
import pandas as pd 
import numpy as np 


def read_data(filename):
    return pd.read_csv(filename)
    
def analyze(data, sections=None):
    available_sections = {
        "head": lambda: data.head(),
        "tail": lambda: data.tail(),
        "info": lambda: data.info(),
        "describe_non": lambda: data.describe(include='O'),
        "describe": lambda: data.describe(),
        
    }

    if sections is None:
        sections = list(available_sections.keys())

    for key in sections:
        if key in available_sections:
            title = key.replace("_", " ").title()
            print(f"\n{'=' * 40} {title} {'=' * 40}\n")
            display(available_sections[key]())


def get_columns(data):
    object_cols = data.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    return {"object": object_cols, "numeric": numeric_cols}


 
def checks_null_values(df):
    '''
    Takes df
    Checks nulls
    '''
    if df.isnull().sum().sum() > 0:
        mask_total = df.isnull().sum().sort_values(ascending=False) 
        total = mask_total[mask_total > 0]

        mask_percent = df.isnull().mean().sort_values(ascending=False) 
        percent = mask_percent[mask_percent > 0] * 100

        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).sort_values(by=['Total'], ascending=False)
    
        return missing_data
    else: 
        print('No NaN found.')