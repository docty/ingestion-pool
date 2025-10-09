import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency

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




 
 
 

def chi_square_test(data, features, plot_expected=False, save_path=None):
    """
    Performs a Chi-Square test of independence on selected categorical/numerical features.
    If the chi-squared value is higher, it suggests a stronger likelihood of a significant connection between the variables.
    If the chi-squared value is higher than the critical value, we will discard the assumption of no relationship.
    A higher chi-square value signifies a greater disparity between observed and expected frequencies.
    Given that the p-value is less than 0.05, we can reject the null hypothesis and conclude that there is indeed a significant association between them.
    
    dof: The degrees of freedom, indicating the number of independent categories in the data
    Parameters:
        data (pd.DataFrame): The dataset.
        features (list): List of column names to include in the contingency table.
        

    Returns:
        dict: A dictionary containing chi2, p-value, degrees of freedom, and expected frequencies.
    """
    # Extract relevant columns
    contingency_data = data[features].iloc[:, :len(features)]

    # Perform Chi-Square test
    chi2, p, dof, expected = chi2_contingency(contingency_data)

    # print("\nChi-Square Test Results:")
    # print(f"Chi2 Statistic: {chi2:.4f}")
    # print(f"Degrees of Freedom: {dof}")
    # print(f"P-value: {p:.6f}")

    # Convert expected frequencies to a DataFrame for better readability
    expected_df = pd.DataFrame(
        expected, 
        index=contingency_data.index if contingency_data.index.size == expected.shape[0] else range(expected.shape[0]),
        columns=contingency_data.columns
    )

    #print("\nExpected Frequencies:")
    #display(expected_df)

    

    return {"chi2": chi2, "p_value": p, "dof": dof, "expected": expected_df}