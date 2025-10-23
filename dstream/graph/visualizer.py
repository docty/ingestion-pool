from .correlation import CorrelationPlot
from .confusion import ConfusionMatrixPlot
from .evaluation import EvaluationPlot
from .timeseries import TimeSeriesPlot
from .decomposition import TimeSeriesDecompositionPlot
from .factory import ChartFactory
 
class Visualizer:
     
    def __init__(self):
        self.chart_factory = ChartFactory()
        self.corr_plot = CorrelationPlot()
        self.conf_plot = ConfusionMatrixPlot()
        self.eval_plot = EvaluationPlot()
        self.ts_plot = TimeSeriesPlot()
        self.ts_decompose = TimeSeriesDecompositionPlot()

    def chart(self, **kwargs):
        self.chart_factory.plot(**kwargs)

    def correlation(self, data):
        self.corr_plot.plot(data)

    def confusion(self, y_test, predictions):
        self.conf_plot.plot(y_test, predictions)

    def evaluation(self, results_df):
        self.eval_plot.plot(results_df)

    def timeseries(self, **kwargs):
        self.ts_plot.plot(**kwargs)

    def decompose(self, **kwargs):
        self.ts_decompose.plot(**kwargs)
