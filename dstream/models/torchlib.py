#import torch.nn as nn


class ScikitWrapper:
    def __init__(self, model):
        #super(ScikitWrapper, self).__init__()
        self.model = model