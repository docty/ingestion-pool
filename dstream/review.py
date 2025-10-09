from IPython.display import display
import pandas as pd 

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


 
def std():
    """ A high standard deviation means that the data is spread out, while a low standard
    deviation means that the data is concentrated around the mean"""
    pass