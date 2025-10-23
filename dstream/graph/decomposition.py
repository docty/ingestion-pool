from .base import BasePlot
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

class TimeSeriesDecompositionPlot(BasePlot):
    

    def plot(self, data, time_col, value_col, model="additive", period=None):
        data = data.copy()
        data.set_index(time_col, inplace=True)
        decomposition = seasonal_decompose(data[value_col], model=model, period=period)

        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        decomposition.observed.plot(ax=axes[0], title="Observed", color="steelblue")
        decomposition.trend.plot(ax=axes[1], title="Trend", color="orange")
        decomposition.seasonal.plot(ax=axes[2], title="Seasonality", color="green")
        decomposition.resid.plot(ax=axes[3], title="Residuals", color="gray")

        plt.tight_layout()
        plt.show()