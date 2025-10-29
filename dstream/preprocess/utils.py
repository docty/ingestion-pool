import pandas as pd

class DataUtils:
    @staticmethod
    def transform(df: pd.DataFrame, column: str, callback):
        df[column] = df[column].apply(callback)
        return df

    @staticmethod
    def replace(df: pd.DataFrame, column: str, old_text: str, new_text: str):
        df[column] = df[column].str.replace(old_text, new_text, regex=False)
        return df

    @staticmethod
    def split(df: pd.DataFrame, column: str, pattern: str, expand: bool = False):
        if expand:
            split_cols = df[column].str.split(pattern, expand=True)
            for i, col in enumerate(split_cols.columns):
                df[f"{column}_{i+1}"] = split_cols[col]
        else:
            df[column] = df[column].str.split(pattern)
        return df

    @staticmethod
    def rename(df: pd.DataFrame, columns: dict):
        return df.rename(columns=columns)

    @staticmethod
    def drop(df: pd.DataFrame, columns: list, axis: int = 1):
        return df.drop(columns=columns, axis=axis)

    @staticmethod
    def drop_duplicates(df: pd.DataFrame, subset=None, keep: str = "first"):
        return df.drop_duplicates(subset=subset, keep=keep)

    @staticmethod
    def concat(dfs: list[pd.DataFrame], axis: int = 0, ignore_index: bool = True):
        return pd.concat(dfs, axis=axis, ignore_index=ignore_index)
