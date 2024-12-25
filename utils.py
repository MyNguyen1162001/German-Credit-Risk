import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder


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


def plot_categorical_violin(
        dataset = pd.DataFrame, 
        cat_column = List[str], 
        num_column = str,
        target_column= str, 
        title='Distribution by Credit Risk'
) -> None:
    for category in cat_column:
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=dataset, 
                    x=category,
                    y= num_column,
                    hue=target_column,
                    split=True,
                    inner='box')
        
        plt.title(f'{num_column} Distribution by {category.title()} and {target_column}')
        plt.xlabel(category.title())
        plt.ylabel(target_column)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()  


def process_categorical_features (
        dataset = pd.DataFrame,
        categories_columns = List[str],
        fill_value  = str ,
        encoding_type = str == 'onehot'
) -> pd.DataFrame: 
    for col in categories_columns: 
        dataset[col] = dataset[col].replace(
                    ['nan', 'None', 'NaN', 'null', '', ' '], 
                    fill_value
                )
        dataset[col] = dataset[col].fillna('missing')

        if encoding_type == 'label':
            encoders = {}
            le = LabelEncoder()
            dataset[col] = le.fit_transform(dataset[col])
        elif encoding_type == 'onehot':
        # Create dummy variables with drop_first=True to avoid multicollinearity
            df_encoded = pd.get_dummies(
                dataset[col], 
                columns=col,
                drop_first=True,
                prefix=col
            )
            dataset = pd.concat([
                dataset.drop(columns=col), 
                df_encoded
            ], axis=1)
    return dataset


def processing_numerical_columns(
        dataset: pd.DataFrame,
        numerical_columns: List[str],
        fill_strategy: str = 'zero',
        handle_outliers: bool = True,
        scaling_type: str = 'standard'
) -> pd.DataFrame: 
    for col in numerical_columns: 
        if fill_strategy == 'zero':
            dataset[col] = dataset[col].fillna(0)
        if fill_strategy == 'mean':
            fill_value = dataset[col].mean()
            dataset[col] = dataset[col].fillna(fill_value)
        if fill_strategy == 'median':
            fill_value = dataset[col].median()
            dataset[col] = dataset[col].fillna(fill_value)
    if handle_outliers:
        for col in numerical_columns:
            Q1 = dataset[col].quantile(0.25)
            Q3 = dataset[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # Cap outliers
            dataset[col] = dataset[col].clip(lower_bound, upper_bound)
    if scaling_type == 'standard':
        scaler = StandardScaler()
    elif scaling_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaling_type == 'robust':
        scaler = RobustScaler()
    return dataset