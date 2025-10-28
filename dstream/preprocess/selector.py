from sklearn.feature_selection import SelectKBest, f_regression, f_classif, VarianceThreshold
from dstream.preprocess.base import  IFeatureSelector
import pandas as pd
from dstream.utils.logged import setLogging
 
logger = setLogging()

class FeatureSelector(IFeatureSelector):
    def __init__(self, k: int = 10, task_type: str = 'regression'):
        self.k = k
        self.selector = SelectKBest(
            score_func=f_regression if task_type == 'regression' else f_classif, k=k
        )
        self.selector2 = VarianceThreshold(threshold=0.0)

    def select(self, X, y):
        logger.info(f"Selecting top {self.k} features...")
        try:
            selected = self.selector.fit_transform(X, y)
            selected_features = X.columns[self.selector.get_support()]
            logger.info(f"Feature selection complete: {list(selected_features)}")
            return pd.DataFrame(selected, columns=selected_features)
        except Exception as e:
            logger.error(f"Error during feature selection: {e}")
            raise