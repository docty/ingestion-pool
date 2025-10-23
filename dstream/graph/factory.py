import matplotlib.pyplot as plt
import seaborn as sns
from .statistical import StatisticalAnnotator
from .base import BasePlot
from scipy import skew, kurtosis


class ChartFactory(BasePlot):
    
    def __init__(self):
        super().__init__()
        self.stats_annotator = StatisticalAnnotator()

    def plot(self, data, x=None, y=None, chart_type='hist', **kwargs):
        chart_type = chart_type.lower()
        method = getattr(self, f"_plot_{chart_type}", None)

        if not method:
            raise ValueError(
                f"Invalid chart type '{chart_type}'. Supported: line, hist, box, scatter, pie, cat, dist, violin."
            )
        method(data, x, y, **kwargs)

    
    def _plot_line(self, data, x, y, ax=None, title=None, xlabel=None, ylabel=None):
        ax = self._init_ax(ax)
        sns.lineplot(data=data, x=x, y=y, ax=ax)
        ax.set_title(title or f"{y} vs {x}")
        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or y)
        self._save_or_show()

    
    def _plot_hist(self, data, x, ax=None, bins=20, title=None):
        ax = self._init_ax(ax)
        x_data = data[x]
        skewness, kurt = skew(x_data), kurtosis(x_data)

        sns.histplot(x_data, bins=bins, kde=True, edgecolor="black", ax=ax)
        ax.set_title(f"{title or x} (Skew: {skewness:.2f}, Kurtosis: {kurt:.2f})")
        ax.set_xlabel(x)
        self._save_or_show()

    
    def _plot_box(self, data, x=None, ax=None, title=None):
        ax = self._init_ax(ax)
        sns.boxplot(data=data[x], ax=ax)
        ax.set_title(title or f"Box Plot of {x}")
        self._save_or_show()

    
    def _plot_scatter(self, data, x, y, ax=None, title=None, annotate_stats=True):
        ax = self._init_ax(ax)
        sns.scatterplot(data=data, x=x, y=y, ax=ax)
        ax.set_title(title or f"{y} vs {x}")
        if annotate_stats:
            self.stats_annotator.correlation_text(ax, x, y, data)
            self.stats_annotator.regression_line(ax, x, y, data)
        self._save_or_show()

    def _plot_regression(self, data, x, y, ax=None, title=None):
        ax = self._init_ax(ax)
        sns.regplot(data=data, x=x, y=y, ci=95, color="teal", ax=ax)
        ax.set_title(title or f"Regression: {y} vs {x}")
        self.stats_annotator.correlation_text(ax, x, y, data)
        self._save_or_show()
    
    def _plot_pie(self, data, x, ax=None, title=None):
        ax = self._init_ax(ax)
        values = data[x].value_counts()
        ax.pie(values, labels=values.index, autopct="%1.1f%%", startangle=90)
        ax.set_title(title or f"Pie Chart of {x}")
        ax.axis("equal")
        self._save_or_show()

    
    def _plot_violin(self, data, x, y, ax=None, title=None):
        ax = self._init_ax(ax)
        sns.violinplot(data=data, x=x, y=y, ax=ax)
        ax.set_title(title or f"Violin Plot of {x} vs {y}")
        self._save_or_show()

