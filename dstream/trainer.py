#from dstream.export import export_model_to_onnx
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import torch.nn as nn
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, mean_squared_error, r2_score
)
import numpy as np


logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',  
    datefmt='%Y-%m-%d %H:%M:%S'   
)

class ScikitWrapper(nn.Module):
    def __init__(self, model):
        super(ScikitWrapper, self).__init__()
        self.model = model

class MLTrainer(Trainer):
    
    def __init__(self, model=None, train_dataset=None, args=None, task_type='classification'):
        """
        Initialize MLTrainer with one or multiple scikit-learn models.

        Args:
            model: A single model instance or dict of models {name: model}
            train_dataset: Dataset or (features, target) tuple
            args: Hugging Face TrainingArguments
        """
        self.is_dict = isinstance(model, dict)
        self.models = {}
        self.task_type = task_type
        
        if self.is_dict:
            
            self.models = {name: ScikitWrapper(m) for name, m in model.items()}
            first_model = next(iter(self.models.values()))
        else:
        
            self.models = {'Model': ScikitWrapper(model)}
            first_model = self.models['Model']

        self.train_dataset = train_dataset
        self.args = args or TrainingArguments(output_dir="./output", report_to="none")
        self.features = None

        super().__init__(model=first_model, train_dataset=train_dataset, args=self.args)

    @classmethod
    def from_classification(cls, train_dataset=None, args: TrainingArguments = TrainingArguments(report_to="none")):
        from dstream import ClassificationModel
        classifier = ClassificationModel.from_default()
        task_type = 'classification'
        return cls(model=classifier.models, train_dataset=train_dataset, args=args, task_type=task_type)

    @classmethod
    def from_regression(cls, train_dataset=None, args: TrainingArguments = TrainingArguments(report_to="none")):
        from dstream import RegressionModel
        regressor = RegressionModel.from_default()
        task_type = 'regression'
        
        return cls(model=regressor.models, train_dataset=train_dataset, args=args, task_type=task_type)


    def train(self):
        """Train one or multiple scikit-learn models."""
        if isinstance(self.train_dataset, Dataset):
            features = self.train_dataset['input']
            target = self.train_dataset['label']
        else:
            features, target = self.train_dataset

        self.features = features

        for name, wrapper in self.models.items():
            logging.info(f"Training model: {name}")
            wrapper.model.fit(features, target)

        logging.info("Training complete.")


    def predict(self, X_test):
        """
        Predict using one or multiple trained models.

        Returns:
            dict or array: If multiple models → dict of predictions;
                           if single model → array of predictions.
        """
        results = {name: wrapper.model.predict(X_test) for name, wrapper in self.models.items()}

       
        return next(iter(results.values())) if not self.is_dict else results

  
    def evaluate(self, eval_dataset=None, metrics=[]):
      """
      Evaluate one or multiple trained models on the provided dataset.

      Args:
          eval_dataset: A Hugging Face Dataset or (X_test, y_test) tuple
          task_type: 'classification' or 'regression'
      
      Returns:
          pd.DataFrame: Evaluation metrics for each model
      """
      if eval_dataset is None:
          raise ValueError("eval_dataset must be provided for evaluation.")

      X_test, y_test = self._prepare_dataset(eval_dataset)

      results = []

       
      for name, wrapper in self.models.items():
          model = wrapper.model
          logging.info(f"Evaluating model: {name}")

          try:
            y_pred = model.predict(X_test)
          except:
            logging.error(f"Prediction failed for {name}: {e}")
            continue
          
          model_metrics = self._evaluate_model(name, model, y_test, y_pred, metrics)
          results.append(model_metrics)

      logging.info("Evaluation complete.")
      return results



    def _prepare_dataset(self, eval_dataset):
        """Extract X_test and y_test from Dataset or tuple."""
        if isinstance(eval_dataset, Dataset):
            X_test, y_test = eval_dataset['input'], eval_dataset['label']
        else:
            X_test, y_test = eval_dataset
        return X_test, y_test
    
    
    def _evaluate_model(self, name, model, y_test, y_pred, custom_metrics=None):
          """Compute metrics for a single model."""
          if self.task_type == 'classification':
              return self._evaluate_classification(name, model, y_test, y_pred, custom_metrics)
          elif self.task_type == 'regression':
              return self._evaluate_regression(name, y_test, y_pred, custom_metrics)
          else:
              raise ValueError("Invalid task_type. Choose 'classification' or 'regression'.")
    
    
    def _evaluate_classification(self, name, model, y_test, y_pred, custom_metrics=None):
          """Evaluate classification model metrics."""
          
          default_metrics = {
              'Accuracy': lambda y_true, y_pred: accuracy_score(y_true, y_pred),
              'Precision': lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0),
              'Recall': lambda y_true, y_pred: recall_score(y_true, y_pred, zero_division=0),
              'F1 Score': lambda y_true, y_pred: f1_score(y_true, y_pred, zero_division=0),
              'Confusion Matrix': lambda y_true, y_pred: confusion_matrix(y_true, y_pred)
          }
    
          metrics = custom_metrics or default_metrics
          return self._compute_metrics(name, metrics, y_test, y_pred)
    
    
    def _evaluate_regression(self, name, y_test, y_pred, custom_metrics=None):
          """Evaluate regression model metrics."""
          default_metrics = {
              'MSE': lambda y_true, y_pred: mean_squared_error(y_true, y_pred),
              'R2 Score': lambda y_true, y_pred: r2_score(y_true, y_pred)
          }
    
          metrics = custom_metrics or default_metrics
          return self._compute_metrics(name, metrics, y_test, y_pred)
    
    
    def _compute_metrics(self, name, metrics_dict, y_test, y_pred):
          """Safely compute metrics and handle errors."""
          results = {'Model': name}
          for metric_name, metric_fn in metrics_dict.items():
              try:
                  results[metric_name] = metric_fn(y_test, y_pred)
              except Exception as e:
                  logging.warning(f"Metric {metric_name} failed for {name}: {e}")
                  results[metric_name] = np.nan
          return results