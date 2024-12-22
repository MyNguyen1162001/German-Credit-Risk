import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns

def visualize_category_features(
    dataset: pd.DataFrame,
    cat_columns: List[str],
) -> None:
    # Calculate number of rows needed (2 plots per row)
    n_rows = (len(cat_columns) + 1) // 2  # Using ceiling division

    # Create subplots
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5*n_rows))
    axes = axes.ravel()  # Flatten axes array for easier indexing

    # Loop through each categorical column
    for idx, col in enumerate(cat_columns):
        sns.countplot(data=dataset, x=col, hue='Risk', ax=axes[idx])
        axes[idx].set_title(f'{col} by Risk')
        axes[idx].tick_params(axis='x', rotation=45)

    # Remove empty subplots if any
    for idx in range(len(cat_columns), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()
    

def visualize_numerical_features(
    dataset: pd.DataFrame,
    num_columns: List[str]
)-> None:
# Create subplots for each numerical column
    fig, axes = plt.subplots(
        len(num_columns), 
        1, 
        figsize=(12, 4*len(num_columns))
    )

    # Loop through each numerical column
    for idx, col in enumerate(num_columns):
        # Create distribution plot using seaborn
        sns.histplot(data=dataset, x=col, hue='Risk', multiple="dodge", ax=axes[idx])
        axes[idx].set_title(f'Distribution of {col} by Risk')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Count')

    plt.tight_layout()
    plt.show()

    # Alternative: Using KDE (Kernel Density Estimation) plots
    plt.figure(figsize=(12, 4*len(num_columns)))

    for idx, col in enumerate(num_columns):
        plt.subplot(len(num_columns), 1, idx+1)
        sns.kdeplot(data=dataset, x=col, hue='Risk',bw_adjust=0.5)
        plt.title(f'Density Distribution of {col} by Risk')
        plt.xlabel(col)
        plt.ylabel('Density')

    plt.tight_layout()
    plt.show()

    # Boxplots for another perspective
    plt.figure(figsize=(12, 4*len(num_columns)))

    for idx, col in enumerate(num_columns):
        plt.subplot(len(num_columns), 1, idx+1)
        sns.boxplot(data=dataset, x='Risk', y=col)
        plt.title(f'Boxplot of {col} by Risk')

    plt.tight_layout()
    plt.show()