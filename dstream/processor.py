from sklearn.manifold import TSNE
import pandas as pd


def tsne(data):
    model = TSNE()
    
    transformed_model = model.fit_transform(data)
    
    xs = transformed_model[:,0]
    ys =transformed_model[:,1]
    return xs, ys

