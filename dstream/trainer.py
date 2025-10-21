#from dstream.export import export_model_to_onnx
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import torch.nn as nn
import logging

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
    
    def __init__(self, model=None, train_dataset=None, args=None):
        """
        Initialize MLTrainer with one or multiple scikit-learn models.

        Args:
            model: A single model instance or dict of models {name: model}
            train_dataset: Dataset or (features, target) tuple
            args: Hugging Face TrainingArguments
        """
        self.is_dict = isinstance(model, dict)
        self.models = {}

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

        return cls(model=classifier.models, train_dataset=train_dataset, args=args)

    @classmethod
    def from_regression(cls, train_dataset=None, args: TrainingArguments = TrainingArguments(report_to="none")):
        from dstream import RegressionModel
        regressor = RegressionModel.from_default()

        return cls(model=regressor.models, train_dataset=train_dataset, args=args)


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

  
    def save(self):
        """Placeholder for ONNX or joblib export."""
        pass
        # export_model_to_onnx(self.models['Model'].model, self.features)
