from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from typing import Tuple, Any
import pandas as pd
from dstream.utils.logged import setLogging

logger = setLogging().getLogger("DimensionalityReducer")


class DimensionalityReducer:
    def __init__(self, method: str = "pca", n_components: int = 2, **kwargs):
        self.method = method.lower()
        self.n_components = n_components

        if self.method == "pca":
            self.reducer = PCA(n_components=n_components, **kwargs)
        elif self.method == "lda":
            self.reducer = LDA(n_components=n_components, **kwargs)
        elif self.method == "tsne":
            self.reducer = TSNE(n_components=n_components, **kwargs)
        else:
            raise ValueError(f"Unsupported reduction method: {self.method}")

    def fit_transform(
        self, data: pd.DataFrame, target: pd.Series = None
    ) -> Tuple[pd.DataFrame, Any]:
        logger.info(f"Applying '{self.method.upper()}' for dimensionality reduction...")
        try:
            if self.method == "lda":
                if target is None:
                    raise ValueError("Target labels are required for LDA.")
                reduced = self.reducer.fit_transform(data, target)
            else:
                reduced = self.reducer.fit_transform(data)

            reduced_df = pd.DataFrame(
                reduced, columns=[f"Component_{i+1}" for i in range(self.n_components)]
            )
            logger.info("Dimensionality reduction complete.")
            return reduced_df, self.reducer
        except Exception as e:
            logger.error(f"Error during dimensionality reduction: {e}")
            raise
