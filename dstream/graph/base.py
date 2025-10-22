import matplotlib.pyplot as plt
import seaborn as sns
from dstream.preprocess.utils import setLogging
 
logger = setLogging()

class BasePlot:
    """Base class for all plots, ensuring consistency in styling and error handling."""
    
    def __init__(self):
        sns.set_style("whitegrid")
    
    def _init_ax(self, ax=None, figsize=(8, 5)):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        return ax

    def _save_or_show(self, save_path=None):
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
