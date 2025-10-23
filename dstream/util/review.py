from IPython.display import display
import pandas as pd
import numpy as np
 
 
class BasicDataAnalyzer:
     
       
    @staticmethod
    def analyze(data: pd.DataFrame, sections):
        sections = sections or [
            "shape", "head", "tail", "info", "describe_non", "describe"
        ]

        available_sections = {
            "shape": lambda: data.shape,
            "head": lambda: data.head(),
            "tail": lambda: data.tail(),
            "info": lambda: data.info(),
            "describe_non": lambda: data.describe(include='O'),
            "describe": lambda: data.describe(),
        }

        for key in sections:
            if key in available_sections:
                title = key.replace("_", " ").title()
                print(f"\n{'=' * 40} {title} {'=' * 40}\n")
                result = available_sections[key]()
                if result is not None:
                    display(result)
  
    @staticmethod
    def nullinspector(data: pd.DataFrame):
        total_missing = data.isnull().sum().sum()

        if total_missing == 0:
            print("No NaN values found.")
            return None

        mask_total = data.isnull().sum().sort_values(ascending=False)
        total = mask_total[mask_total > 0]

        mask_percent = data.isnull().mean().sort_values(ascending=False)
        percent = mask_percent[mask_percent > 0] * 100

        missing_data = pd.concat(
            [total, percent],
            axis=1,
            keys=['Total', 'Percent']
        ).sort_values(by=['Total'], ascending=False)

        print(f"Found {int(total_missing)} missing values.")
        display(missing_data)

    @staticmethod
    def duplicate(data: pd.DataFrame):
        print("################## DataFrame Duplicated ###########\n")
        duplicate_count = data.duplicated().sum()
        display(duplicate_count)
