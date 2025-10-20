from transformers import Trainer, TrainingArguments 
from datasets import Dataset
import torch.nn as nn

class ScikitWrapper(nn.Module):

  def __init__(self, model):
    super(ScikitWrapper, self).__init__()
    self.model = model

class MLTrainer(Trainer):
  
  @classmethod
  def from_scikit(cls, model, train_dataset=None, args: TrainingArguments = TrainingArguments(report_to='none')):
    wrapmodel = ScikitWrapper(model)
    return cls(model=wrapmodel,  train_dataset= train_dataset, args=args)

  def train(self):
    if isinstance(self.train_dataset, Dataset):
      features = self.train_dataset['input']
      target = self.train_dataset['label']
    else:
      features, target = self.train_dataset

    self.model.model.fit(features, target)


  def predict(self, X_test):
    return self.model.model.predict(X_test)