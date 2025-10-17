import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
 

def plot_chart(
    data=None,
    x=None,
    y=None,
    chart_type='hist',
    xlabel=None,
    ylabel=None,
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
         
        sns.lineplot(data=data,x=x,y=y, ax=ax)
        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or x)
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

    elif chart_type == 'cat':
        if data is None or x is None:
            raise ValueError("Scatter plot requires 'data', 'x'.")
        sns.catplot(data=data, x=x, kind='count',aspect=2.4, ax=ax)
        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or y)
        ax.set_title(title)

    elif chart_type == 'dist':
        if data is None or x is None:
            raise ValueError("Scatter plot requires 'data', 'x'.")
        sns.displot(data=data, x=x, kde=True, bins=25)
        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or y)
        ax.set_title(title)

    elif chart_type == 'violin':
        if data is None or x is None:
            raise ValueError("Scatter plot requires 'data', 'x'.")
        sns.violinplot(data=data, x=x, y=y)
        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or y)
        ax.set_title(title)

     
    else:
        raise ValueError("Invalid chart_type. Choose from: 'line', 'pie', 'hist', 'box', 'scatter'.")

    
    if save_path and ax is None:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

     
    if ax is None:
        plt.show()


def histogram_plot(df, numerical_columns):
    '''
    Takes df, numerical columns as list
    Returns a group of histagrams
    '''
    f = pd.melt(df, value_vars=numerical_columns) 
    g = sns.FacetGrid(f, col='variable',  col_wrap=4, sharex=False, sharey=False)
    g = g.map(sns.distplot, 'value')
    return g

def correlate(data): 
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()


def heatmap_plot(df, dependent_variable):
    '''
    Takes df, a dependant variable as str
    Returns a heatmap of all independent variables' correlations with dependent variable 
    '''
    plt.figure(figsize=(8, 10))
    g = sns.heatmap(df.corr()[[dependent_variable]].sort_values(by=dependent_variable), 
                    annot=True, 
                    cmap='coolwarm', 
                    vmin=-1,
                    vmax=1) 
    return g

def corr_list(df):

      return  (df.corr()
          .unstack()
          .sort_values(kind="quicksort",ascending=False)
          .drop_duplicates().iloc[1:]); df_out

    
def box_plot(df, col_name, title=None, xlabel=None):
    """
    Draw's a single horizontal boxplot
    Parameters
    ----------
    df : Pandas Data Frame
    col_name : column name in data frame
    title : Plot title
    xlabel : X-axis label
    ylabel : Y-axis label
    """
    fig = plt.figure(figsize=(10, 6))  # define plot area
    ax = fig.add_subplot(111)  # add single subplot
    sns.boxplot(df[col_name], ax=ax)  # Use seaborn plot
    if not title:
        title = 'Boxplot of {}'.format(col_name)
    ax.set_title(title)  # Give the plot a main title
    if not xlabel:
        xlabel = col_name
    ax.set_xlabel(xlabel)  # Set text for the x axis


def bar_plot(df, col_name, title=None, xlabel=None, ylabel='Count'):
    """
    Draw's a single bar plot
    Parameters
    ----------
    df : Pandas Data Frame
    col_name : column name in data frame
    title : Plot title
    xlabel : X-axis label
    ylabel : Y-axis label
    """
    fig = plt.figure(figsize=(10, 6))  # define plot area
    ax = fig.add_subplot(111)  # add single subplot
    ax = df[col_name].value_counts().plot.bar(
        color='steelblue')  # Use pandas bar plot
    if not title:
        title = 'Barplot of {}'.format(col_name)
    ax.set_title(title)  # Give the plot a main title
    if not xlabel:
        xlabel = f'No. of {col_name}'
    ax.set_xlabel(xlabel)  # Set text for the x axis
    ax.set_ylabel(ylabel)  # Set text for the y axis

def pairplot(data):
    sns.set_style('darkgrid')
    sns.pairplot(data, kind='reg', diag_kind='kde',
             plot_kws={'line_kws':{'color':'red'}}, diag_kws={'color':'green'})

def lmplot(data, x, y, hue="classification"):
    sns.lmplot(data=df,x="pH",y="Conductivity (µS/cm)",hue=hue)
    plt.title(f"Relation between {x} and the {y}")
    plt.show()

def jointplot(data, x, y, hue="classification"):
    g = sns.jointplot(data=data,x=x,y=y,hue=hue,height=8)
    g.plot_joint(sns.kdeplot,color='y',zorder=0)
    g.plot_marginals(sns.rugplot,color='r',height=-0.2,clip_on=False)
    plt.show()

def regplot(data, x, y):
    sns.regplot(data=data, y=y, x=x)
    plt.show()