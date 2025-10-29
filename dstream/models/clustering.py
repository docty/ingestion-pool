from dstream.models.base import BaseModelRegistry
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN,   SpectralClustering


class ClusteringModel(BaseModelRegistry):
    def __init__(self, models=None):
        self.models = models or {}

    @classmethod
    def from_default(cls):
        default_models = {
            "KMeans": KMeans(n_clusters=3, random_state=0, n_init=10),
            "Agglomerative Clustering": AgglomerativeClustering(n_clusters=3),
            "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
            "Spectral Clustering": SpectralClustering(n_clusters=3, affinity="nearest_neighbors", random_state=0),
        }
        return cls(models=default_models)

    def add_model(self, name, model):
        self.models[name] = model

    @property
    def summary(self):
        return {name: type(model).__name__ for name, model in self.models.items()}
