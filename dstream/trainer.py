from transformers import Trainer 

class MLTrainer(Trainer):
  def __init__(self, model=None, train_dataset=None):
    self.model = model
    self.train_dataset = train_dataset

  def train(self):
    features, target = self.train_dataset
    self.model.fit(features, target)

  def predict(self, X_test):
    return self.model.predict(X_test)