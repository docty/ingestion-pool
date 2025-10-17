import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
 

def plot_chart(
    x=None,
    y=None,
    data=None,
    chart_type='hist',
    xlabel='x',
    ylabel='y',
    title=None,
    save_path=None,
    bins=20,
    ax=None,
    features=None
):
    """
    General-purpose plotting function supporting:
    - 'line'        : Line chart
    - 'pie'         : Pie chart
    - 'hist'        : Histogram
    - 'box'         : Box plot
    - 'scatter'     : Scatter plot
    
    Works either standalone or within a subplot (via ax).
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    title = title or "Data Visualization"

    chart_type = chart_type.lower()

    
    if chart_type == 'line':
        if x is None or y is None:
            raise ValueError("Line chart requires both x and y data.")
        xdata = data[x]
        ydata = data[y]
        
        ax.plot(xdata, ydata, marker='o', linestyle='-', linewidth=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.6)

    
    elif chart_type == 'pie':
        if x is None or y is None:
            raise ValueError("Pie chart requires both x and y data.")
        xdata = data[x]
        
        
        ax.pie(xdata, labels=x, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title(title)

    
    elif chart_type == 'hist':
        if x is None:
            raise ValueError("Histogram requires y data.")
        x_data = data[x]
        
        skewness = skew(x_data)
        kurto = kurtosis(x_data)

        sns.histplot(x_data, bins=bins, edgecolor='black', kde=True, alpha=0.75, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title}, Skew: {skewness:.5f}, Kurtosis: {kurto:.5f}')
        ax.grid(axis='y', linestyle='--', alpha=0.6)

   
    elif chart_type == 'box':
        if data is None or features is None:
            raise ValueError("Box plot requires 'data' and 'features'.")
        sns.boxplot(data=data[x], ax=ax)
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45)

    
    elif chart_type == 'scatter':
        if data is None or x is None or y is None:
            raise ValueError("Scatter plot requires 'data', 'x', and 'y'.")
        sns.scatterplot(data=data, x=x, y=y, ax=ax)
        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or y)
        ax.set_title(title)

    

     
    else:
        raise ValueError("Invalid chart_type. Choose from: 'line', 'pie', 'hist', 'box', 'scatter'.")

    
    if save_path and ax is None:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

     
    if ax is None:
        plt.show()