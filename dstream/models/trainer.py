#from dstream.export import export_model_to_onnx
from dstream.models.util import TaskType
import numpy as np 
from dstream.utils.logged import setLogging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score,  mean_squared_error, r2_score
)
from dstream.models.trainargs import TrainingArguments

logging = setLogging().getLogger("Trainer")


class MLTrainer:
    
    def __init__(self, model, train_dataset=None, 
                eval_dataset=None, args:TrainingArguments=TrainingArguments, 
                task_type: TaskType = TaskType.CLASSIFICATION
                ):
         
        self.is_dict = isinstance(model, dict)
        self.models = {}
        self.task_type = task_type
         
        if self.is_dict:   
            self.models = {model_name: m for model_name, m in model.items()} #{name: ScikitWrapper(m) for name, m in model.items()}
        else:
            self.models = {'Model': model} #{'Model': ScikitWrapper(model)}
            
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.args = args
        

    @classmethod
    def from_classification(cls, train_dataset=None,  eval_dataset=None, args: TrainingArguments = TrainingArguments):
        from dstream import ClassificationModel
        classifier = ClassificationModel.from_default()
        task_type = TaskType.CLASSIFICATION

        return cls(model=classifier.models, train_dataset=train_dataset, eval_dataset=eval_dataset, args=args, task_type=task_type)

    @classmethod
    def from_regression(cls, train_dataset=None,  eval_dataset=None, args: TrainingArguments = TrainingArguments):
        from dstream import RegressionModel
        regressor = RegressionModel.from_default()
        task_type = TaskType.REGRESSION
        
        return cls(model=regressor.models, train_dataset=train_dataset, eval_dataset=eval_dataset, args=args, task_type=task_type)


    def train(self):
        # if isinstance(self.train_dataset, Dataset):
        #     features = self.train_dataset['input']
        #     target = self.train_dataset['label']
        # else:
        #     features, target = self.train_dataset
        features, target = self.train_dataset

        self.features = features

        for name, wrapper in self.models.items():
            logging.info(f"Model: {name}")
            wrapper.fit(features, target)

        logging.info("Training complete.")
        

    def predict(self, X_test):

        results = {name: wrapper.predict(X_test) for name, wrapper in self.models.items()}
        return results
        #return next(iter(results.values())) if not self.is_dict else results

  
    def evaluate(self, eval_dataset=None, metrics=[]):
      
      if eval_dataset is None:
          raise ValueError("eval_dataset must be provided for evaluation.")

      X_test, y_test = self._prepare_dataset(eval_dataset)

      results = []

       
      for name, wrapper in self.models.items():
          model = wrapper
          logging.info(f"Evaluating model: {name}")

          try:
            y_pred = model.predict(X_test)
          except Exception as e:
            logging.error(f"Prediction failed for {name}: {e}")
            continue
          
          model_metrics = self._evaluate_model(name, model, y_test, y_pred, metrics)
          results.append(model_metrics)

      logging.info("Evaluation complete.")
      return results



    def _prepare_dataset(self, eval_dataset):
        """Extract X_test and y_test from Dataset or tuple."""
        # if isinstance(eval_dataset, Dataset):
        #     X_test, y_test = eval_dataset['input'], eval_dataset['label']
        # else:
        #     X_test, y_test = eval_dataset
        X_test, y_test = eval_dataset
        return X_test, y_test
    
    
    def _evaluate_model(self, name, model, y_test, y_pred, custom_metrics=None):
          """Compute metrics for a single model."""
          if self.task_type == TaskType.CLASSIFICATION:
              return self._evaluate_classification(name, model, y_test, y_pred, custom_metrics)
          elif self.task_type == TaskType.REGRESSION:
              return self._evaluate_regression(name, y_test, y_pred, custom_metrics)
          else:
              raise ValueError("Invalid task_type. Choose 'Classification' or 'Regression'.")
    
    
    def _evaluate_classification(self, name, model, y_test, y_pred, custom_metrics=None):
          """Evaluate classification model metrics."""
          
          default_metrics = {
              'Accuracy': lambda y_true, y_pred: accuracy_score(y_true, y_pred),
              'Precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
              'Recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted'),
              'F1 Score': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
             
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

    def save(self):
        import joblib 
        import os 
        os.makedirs(self.args.output_dir, exist_ok=True)
        for name, wrapper in self.models.items():
            joblib.dump(wrapper, f'{self.args.output_dir}/{name}.pkl')
        logging.info(f"Model saved successfully in {self.args.output_dir}")