import seaborn as sns
from .base import BasePlot

class CorrelationPlot(BasePlot):
    """correlation visualization."""
    
    def plot(self, data, annot=True):
        ax = self._init_ax()
        sns.heatmap(data.corr(), annot=annot, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Matrix")
        self._save_or_show()