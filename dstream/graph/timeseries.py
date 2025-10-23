from .base import BasePlot
import matplotlib.pyplot as plt
import seaborn as sns

class TimeSeriesPlot(BasePlot):
    """Time series trend visualization with optional anomaly detection."""

    def plot(self, data, time_col, value_col, window=5, anomaly_std=2, show_anomalies=True):
        ax = self._init_ax(figsize=(12, 6))

        # Compute rolling mean
        data = data.copy()
        data["RollingMean"] = data[value_col].rolling(window=window).mean()

        # Detect anomalies
        mean = data[value_col].mean()
        std = data[value_col].std()
        upper_bound = mean + anomaly_std * std
        lower_bound = mean - anomaly_std * std

        sns.lineplot(data=data, x=time_col, y=value_col, label="Actual", ax=ax)
        sns.lineplot(data=data, x=time_col, y="RollingMean", label=f"{window}-Point Moving Avg", ax=ax, color="orange")

        # Highlight anomalies
        if show_anomalies:
            anomalies = data[(data[value_col] > upper_bound) | (data[value_col] < lower_bound)]
            ax.scatter(anomalies[time_col], anomalies[value_col], color="red", label="Anomalies", zorder=5)

        ax.axhline(upper_bound, color="gray", linestyle="--", alpha=0.6)
        ax.axhline(lower_bound, color="gray", linestyle="--", alpha=0.6)
        ax.set_title(f"Time Series: {value_col} Over Time")
        ax.legend()
        self._save_or_show()

