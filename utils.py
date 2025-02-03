import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

import matplotlib.pyplot as plt
import matplotlib
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
        dataset : pd.DataFrame, 
        cat_column : List[str], 
        num_column : str,
        target_column: str, 
        title = 'Distribution by Credit Risk',
        # colors : List[str] = ['#91c08b', '#8080ff']
) -> None:
    for category in cat_column:
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=dataset, 
                    x=category,
                    y= num_column,
                    hue=target_column,
                    split=True,
                    inner='box',
                    linewidth=1.5,       
                    saturation=0.75)

        plt.title(f'{num_column} Distribution by {category.title()} and {target_column}')
        plt.xlabel(category.title())
        plt.ylabel(target_column)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()  


def plot_categorical_violin_not_default(
        dataset : pd.DataFrame, 
        cat_column : List[str], 
        num_column : str,
        target_column: str, 
        title = 'Distribution by Credit Risk',
        colors : List[str] = ['#91c08b', '#8080ff']
) -> None:
    for category in cat_column:
        plt.figure(figsize=(12, 6))
        ax = sns.violinplot(data=dataset, 
                    x=category,
                    y= num_column,
                    hue=target_column,
                    split=True,
                    inner='box',
                    palette=colors,
                    linewidth=1.5,       
                    saturation=1)
        
        for violin in ax.collections:
            if isinstance(violin, matplotlib.collections.PolyCollection):
                violin.set_edgecolor(violin.get_facecolor())

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


import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from typing import Tuple, Union
from sklearn.base import BaseEstimator 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def plot_threshold_analysis(
        model: BaseEstimator, 
        X_test: Union[pd.DataFrame, np.ndarray], 
        y_test: Union[pd.DataFrame, np.ndarray], 
        target_label: int =1
) -> None :
    y_prob = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0, 1, 0.01)
    aucs = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        auc = roc_auc_score(y_test, y_pred)
        aucs.append(auc)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, aucs, 'b-')
    plt.xlabel('Threshold')
    plt.ylabel('AUC')
    plt.title('AUC vs Classification Threshold')
    plt.grid(True)
    plt.legend()
    plt.show()


def find_optimal_threshold(
    model: BaseEstimator,          # scikit-learn model or pipeline
    X_test: Union[pd.DataFrame, np.ndarray],  # Features can be DataFrame or numpy array
    y_test: Union[pd.Series, np.ndarray],     # Target can be Series or numpy array
    current_accuracy: float = 0.75  # Float between 0 and 1
) -> Tuple[float, float]: 
    # Get probability predictions
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1
    
    # Create range of thresholds
    thresholds = np.arange(0.1, 0.9, 0.01)
    
    # Store metrics for each threshold
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    # Calculate metrics for each threshold
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype('int')
        
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
    
    # Plot the metrics
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies, label='Accuracy')
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, f1_scores, label='F1 Score')
    plt.axhline(y=current_accuracy, color='r', linestyle='--', label='Current Accuracy')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Find threshold that maximizes accuracy
    best_accuracy_idx = np.argmax(accuracies)
    best_threshold_accuracy = thresholds[best_accuracy_idx]
    
    # Find threshold that maximizes F1 score
    best_f1_idx = np.argmax(f1_scores)
    best_threshold_f1 = thresholds[best_f1_idx]
    
    print(f"Best threshold for accuracy: {best_threshold_accuracy:.3f}")
    print(f"Best accuracy score: {accuracies[best_accuracy_idx]:.3f}")
    print(f"\nBest threshold for F1: {best_threshold_f1:.3f}")
    print(f"Best F1 score: {f1_scores[best_f1_idx]:.3f}")
    
    return best_threshold_accuracy, best_threshold_f1


# Sklearn imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from typing import Dict, Union, Tuple, TypeVar
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Boosting models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def create_model_pipeline(
    model_name: str ='lr'
) -> Pipeline:
   models = {
       'lr': LogisticRegression(random_state=42),
       'rf': RandomForestClassifier(random_state=42),
        'xgb': XGBClassifier(
        use_label_encoder=False,  # Add this
        eval_metric='logloss',    # Add this
        random_state=42,
        enable_categorical=True    # Add this if you have categorical features
        ),
       'lgb': LGBMClassifier(random_state=42)
   }
   
   pipeline = Pipeline([
       ('scaler', StandardScaler()),
       ('model', models[model_name])
   ])
   
   return pipeline


def train_evaluate(
        X_train: Union[pd.DataFrame, np.ndarray], 
        X_test: Union[pd.DataFrame, np.ndarray], 
        y_train: Union[pd.DataFrame, np.ndarray], 
        y_test: Union[pd.DataFrame, np.ndarray], 
        model: BaseEstimator,
        threshold: float 
    ) -> Tuple[Dict[str, float], BaseEstimator]:
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:,1]
    y_pred = (y_prob >= threshold).astype('int')
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob)
    }
    
    return metrics, model


def cross_validate_models(
      X:Union[pd.DataFrame, np.ndarray], 
      y: Union[pd.DataFrame, np.ndarray],
      models: List[str] =['lr', 'rf', 'lgb']
) -> Dict[str, Dict[str, float]]:
   results: Dict[str, Dict[str, float]] = {}
   for model_name in models:
       pipeline = create_model_pipeline(model_name)
       scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
       results[model_name] = {
           'mean_auc': scores.mean(),
           'std_auc': scores.std()
       }
   return results


from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def split_train_test(
    dataset: pd.DataFrame,
    target_variable: str,
    test_size: float = 0.2,
    is_rebalance: bool = True,
    positive_num: int = None,
    negative_num: int = None,
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    # Split features and target
    X = dataset.drop(target_variable, axis=1)
    y = dataset[target_variable]
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=test_size, 
        random_state=42, 
        stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # print(y_train.value_counts())

    if is_rebalance:
        smote = SMOTE(
            sampling_strategy={1: positive_num, 0: negative_num},
            random_state=42
        )
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        print(y_train.value_counts())
        # Convert y_train to numpy array for consistency
        # y_train = np.array(y_train)

    return X_train_scaled, X_test_scaled, y_train, y_test


def plot_credit_boxplot(
    dataset: pd.DataFrame,
    cat_column: List[str],
    num_column: str,
    target_column: str,
    colors: List[str] = ['#91c08b', '#ff9999'],  # Light green for good, light red for bad
    figsize_length : int = 12,
    figsize_width: int = 6
) -> None:
    plt.figure(figsize=(figsize_length, figsize_width))
    
    # Create boxplot with customization
    for category in cat_column:
        sns.boxplot(data=dataset,
                    x=category,
                    y=num_column,
                    hue=target_column,
                    palette=colors,
                    flierprops={'marker': 'o', 'markerfacecolor': None, 'markersize': 4},
                    boxprops={'alpha': 0.5},
                    whiskerprops={'linestyle': '-'},
                    medianprops={'color': 'black'},
                    showfliers=True)  # Show outlier points
        
        # Customize the plot
        plt.title(f'{num_column} Distribution by {category} and Credit Risk')
        plt.xlabel(f'{category}')
        plt.ylabel('Credit Amount (US Dollar)')
        
        # Customize legend
        plt.legend(title='', labels=['good', 'bad'], 
                bbox_to_anchor=(1, 1), loc='upper right')
        
        # Add gridlines
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def ScoreScalingParameters (
        points_to_double_odds=20, 
        ref_score=600, 
        ref_odds=50
):
    factor = points_to_double_odds / np.log(2)
    offset = ref_score - factor * np.log(ref_odds)
    return factor, offset


def calculate_woe_iv(
        X: pd.DataFrame,
        y: pd.DataFrame,
        features, 
        bins=10
):  
      
    bins_dict = {}
    woe_dict = {}
    iv_dict = {}
    for feature in features: 
        if X[feature].dtypes in ['object', 'category']:
            # For categorical variables, use unique values as bins
            groups = pd.qcut(X[feature].astype('category').cat.codes, q=bins, duplicates='drop')
        else:
            # For numerical variables, create equal-frequency bins
            groups = pd.qcut(X[feature], q=bins, duplicates='drop')
        
        # Store bin edges for future use
        bins_dict[feature] = groups.unique()
        grouped = pd.DataFrame({'group': groups, 'target': y}).groupby('group')
        iv = 0
        
        for group in grouped.groups.keys():
            group_stats = grouped.get_group(group)
            good = sum(group_stats['target'] == 0)
            bad = sum(group_stats['target'] == 1)
            
            # Add smoothing to handle zero counts
            good = good + 0.5
            bad = bad + 0.5
            
            good_rate = good / (sum(y == 0))
            bad_rate = bad / (sum(y == 1))
            
            woe = np.log(good_rate / bad_rate)
            iv += (good_rate - bad_rate) * woe
            
            woe_dict[group] = woe
        
        woe_dict[feature] = woe_dict
        iv_dict[feature] = iv
        
    return woe_dict, iv_dict


def transform_woe(
        woe_dict: pd.DataFrame,
        X: pd.DataFrame, 
        feature: str
):
    if X[feature].dtype in ['object', 'category']:
        groups = pd.qcut(X[feature].astype('category').cat.codes, 
                        q=len(bins_dict[feature]), 
                        duplicates='drop')
    else:
        groups = pd.qcut(X[feature], 
                        q=len(bins_dict[feature]), 
                        duplicates='drop')
    
    return groups.map(woe_dict[feature])


def fit(self, X, y):
    """Fit the scorecard model"""
    # Calculate WoE and IV for all features
    X_woe = pd.DataFrame()
    for feature in X.columns:
        calculate_woe_iv(X, y, feature)
        X_woe[feature] = transform_woe(X, feature)
    
    # Fit logistic regression
    model = LogisticRegression(random_state=42)
    model.fit(X_woe, y)
    
    return self


def transform_to_score(model, X, offset, factor):
    """Transform features to credit score"""
    X_woe = pd.DataFrame()
    for feature in X.columns:
        X_woe[feature] = transform_woe(X, feature)
    
    # Calculate log odds
    log_odds = model.predict_proba(X_woe)[:, 1]
    log_odds = np.log(log_odds / (1 - log_odds))
    
    # Transform to score
    scores = offset + factor * log_odds
    return scores





    