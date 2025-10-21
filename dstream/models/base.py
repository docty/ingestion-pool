from abc import ABC, abstractmethod

 
class BaseModelRegistry(ABC):
    @abstractmethod
    def summary(self):
        pass

    @abstractmethod
    def add_model(self, name, model):
        pass
