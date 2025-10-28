from .base import BaseModelRegistry
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

class RegressionModel(BaseModelRegistry):
     
    def __init__(self, models=None):
         
        self.models = models or {}

    @classmethod
    def from_default(cls):
        
        default_models = {
            'Linear Regression': LinearRegression(n_jobs=-1),
            'Ridge Regression': Ridge(alpha=1.0, random_state=0),
            'Lasso Regression': Lasso(alpha=0.1, random_state=0),
            'Decision Tree Regressor': DecisionTreeRegressor(max_depth=5, random_state=0),
            'Random Forest Regressor': RandomForestRegressor(
                n_estimators=200, min_samples_leaf=0.1, random_state=0, n_jobs=-1
            ),
            'SVR': SVR(kernel='rbf'),
            'XGBoost Regressor': XGBRegressor(
                n_estimators=150, learning_rate=0.05,
                max_depth=6, n_jobs=5, random_state=0
            )
        }
        return cls(models=default_models)

    def add_model(self, name, model):
        self.models[name] = model

   
    @property
    def summary(self):
        return {name: type(model).__name__ for name, model in self.models.items()}