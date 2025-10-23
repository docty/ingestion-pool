from .base import BasePlot
import matplotlib.pyplot as plt

class EvaluationPlot(BasePlot):
    
    def plot(self, results_df):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Model Performance Metrics", fontsize=18, fontweight="bold")
        
        metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

        for ax, metric in zip(axes.flatten(), metrics):
            sns.barplot(x="Model", y=metric, data=results_df, color="orange", ax=ax)
            ax.set_title(metric)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        self._save_or_show()

