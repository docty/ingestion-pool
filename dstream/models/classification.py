from .base import BaseModelRegistry
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

class ClassificationModel(BaseModelRegistry):
     
    def __init__(self, models=None):
         
        self._models = models or {}

    @classmethod
    def from_default(cls):
        
        default_models = {
            'Logistic Regression': LogisticRegression(max_iter=120, n_jobs=20),
            'Decision Tree': DecisionTreeClassifier(max_depth=4),
            'Random Forest Classifier': RandomForestClassifier(
                n_estimators=300, min_samples_leaf=0.16
            ),
            'K Neighbors': KNeighborsClassifier(n_neighbors=9, leaf_size=20),
            'SVM': SVC(kernel='rbf', probability=True),
            'XGBoost': XGBClassifier(
                max_depth=8, n_estimators=125,
                learning_rate=0.03, n_jobs=5
            )
        }
        return cls(models=default_models)

    def add_model(self, name, model):
        self._models[name] = model

   
    @property
    def summary(self):
        return {name: type(model).__name__ for name, model in self._models.items()}