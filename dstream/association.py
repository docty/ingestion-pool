import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def analyze_relationships(data, features, matrix_type='corr', plot_heatmap=False, save_path=None):
    """
    Analyzes relationships between selected numerical features by computing
    either a correlation or covariance matrix, with optional heatmap visualization.

    Parameters:
        data (pd.DataFrame): Input dataset
        features (list): List of numerical columns
        matrix_type (str): 'corr' for correlation matrix or 'cov' for covariance matrix
        plot_heatmap (bool): If True, display a heatmap of the matrix
        save_path (str): Optional path to save the heatmap
    """

    # A correlation coefficient of +1 denotes highpositive correlation, indicating that as one 
    #feature increases, the other also increases, and vice versa. Conversely, a coefficient
    #of -1 signifies high negative correlation, suggesting that as one feature increases, the 
    #other decreases, and vice versa 

    # Covariance matrix
    # A high positive covariance indicates that both variables move in the same direction as
    # one increases, the other tends to increase and vice versa. Conversely, a high
    # negative covariance implies that both variables move in opposite directions as one
    # increases, the other tends to decrease, and vice versa. 

    if matrix_type not in ['corr', 'cov']:
        raise ValueError("Invalid matrix_type. Choose 'corr' or 'cov'.")

     
    if matrix_type == 'corr':
        matrix = data[features].corr()
        print("\nCorrelation Matrix:")
    else:
        matrix = data[features].cov()
        print("\nCovariance Matrix:")

    if plot_heatmap:
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True,  fmt=".5f", square=True)
        plt.title(f"{matrix_type.capitalize()} Heatmap")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        
        plt.show()

    return matrix