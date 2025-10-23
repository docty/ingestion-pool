import seaborn as sns
from .base import BasePlot
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class ConfusionMatrixPlot(BasePlot):
    """Plots multiple confusion matrices for given models."""

    def plot(self, y_test, predictions_dict):
        num_models = len(predictions_dict)
        ncols = 2
        nrows = int(np.ceil(num_models / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows))
        axes = np.array(axes).flatten()

        for ax, (model_name, y_pred) in zip(axes, predictions_dict.items()):
            cm = confusion_matrix(y_test, y_pred)
            cm_norm = cm / np.sum(cm)
            sns.heatmap(
                cm_norm, annot=True, fmt=".2%", cmap="Reds", ax=ax, cbar=True, square=True
            )
            ax.set_title(f"{model_name} Confusion Matrix", fontsize=14)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")

        for extra_ax in axes[len(predictions_dict):]:
            extra_ax.axis("off")

        plt.tight_layout()
        self._save_or_show()


