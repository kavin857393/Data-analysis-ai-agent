from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import re
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    OneHotEncoder, 
    OrdinalEncoder, 
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PowerTransformer
)
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression


def feature_engineering(df, llm, target_column, max_encoded_columns=10, outlier_threshold=3):
    """Perform comprehensive feature engineering on the dataset.
    
    Args:
        df: Input DataFrame
        llm: Language model instance for guidance
        target_column: Name of the target column
        max_encoded_columns: Maximum number of columns for one-hot encoding
        outlier_threshold: Z-score threshold for outlier detection
        
    Returns:
        tuple: (engineered_df, feature_engineering_summary)
    """


    # Input validation
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    if df[target_column].isnull().all():
        raise ValueError(f"Target column '{target_column}' contains only null values")
        
    # Initialize feature engineering summary
    feature_engineering_summary = {
        "transformations": [],
        "new_features": [],
        "feature_selection": {},
        "dimensionality_reduction": {}
    }


    # 1. Prompt the LLM for Feature Engineering Strategy
    feature_engineering_template = """
    Task: Comprehensive Feature Engineering Strategy

    Data: {df_head}
    Target: {target_column}
    Shape: {df.shape}
    Columns: {', '.join(df.columns)}
    
    Instructions:
    1. Suggest comprehensive feature engineering strategies including:
        - Advanced imputation methods (KNN, iterative)
        - Outlier detection and handling
        - Feature encoding strategies
        - Feature scaling and normalization
        - Feature interaction and polynomial features
        - Dimensionality reduction techniques
        - Feature selection methods
    2. Consider the target column '{target_column}' when making suggestions.
    3. Provide specific parameters for each technique when applicable.
    4. Suggest feature selection methods based on feature importance.


    Example Response:
    ## Feature Engineering Strategy:

    **Imputation:**
    - Numerical features: median
    - Categorical features: most_frequent

    **Outlier Removal:**
    - Method: z-score
    - Threshold: 3

    **Encoding:**
    - Categorical features with less than 10 unique values: one-hot encoding
    - Categorical features with 10 or more unique values: ordinal encoding

    **New Features:**
    - Create a new feature by combining feature A and feature B.
    - Ensure new features don't leak target information.
    """

    feature_engineering_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a feature engineering expert."),
        HumanMessage(content=feature_engineering_template)
    ])

    response = llm.invoke(feature_engineering_prompt.format_messages(
        df_head=df.head().to_json(), target_column=target_column
    ))
    llm_response = getattr(response, "content", None)
    if not llm_response:
        raise ValueError("Invalid response from LLM: Missing content attribute")

    # 2. Parse LLM Response and Extract Strategies
    imputation_strategies = {}
    outlier_removal_strategy = {}
    encoding_strategies = {}
    new_features_suggestions = []

    # Allowed strategies for imputation.
    valid_strategies = {"mean", "median", "most_frequent", "constant"}

    # Imputation Strategies for numerical features
    num_match = re.search(r"Numerical features:\s*(?:Use\s+)?(\w+)", llm_response, re.IGNORECASE)
    if num_match:
        num_strategy = num_match.group(1).lower()
        if num_strategy not in valid_strategies:
            num_strategy = "median"
        imputation_strategies['numerical'] = num_strategy

    # Imputation Strategies for categorical features
    cat_match = re.search(r"Categorical features:\s*(?:Use\s+)?(\w+)", llm_response, re.IGNORECASE)
    if cat_match:
        cat_strategy = cat_match.group(1).lower()
        if cat_strategy not in valid_strategies:
            cat_strategy = "most_frequent"
        imputation_strategies['categorical'] = cat_strategy

    # Outlier Removal Strategy
    match = re.search(r"Method:\s*(\w+)", llm_response, re.IGNORECASE)
    if match:
        outlier_removal_strategy['method'] = match.group(1).lower()
    match = re.search(r"Threshold:\s*(\d+)", llm_response)
    if match:
        outlier_removal_strategy['threshold'] = int(match.group(1))

    # Encoding Strategies
    if "one-hot encoding" in llm_response.lower():
        encoding_strategies['low_cardinality'] = 'one-hot'
    if "ordinal encoding" in llm_response.lower():
        encoding_strategies['high_cardinality'] = 'ordinal'

    # New Features Suggestions
    for line in llm_response.split('\n'):
        if "Create a new feature" in line:
            new_features_suggestions.append(line.strip())

    # 3. Imputation
    numeric_features = [col for col in df.select_dtypes(include=np.number).columns if col != target_column]
    categorical_features = [col for col in df.select_dtypes(exclude=np.number).columns if col != target_column]

    if imputation_strategies:
        num_imputer = SimpleImputer(strategy=imputation_strategies.get('numerical', 'mean'))
        cat_imputer = SimpleImputer(strategy=imputation_strategies.get('categorical', 'most_frequent'))

        df[numeric_features] = num_imputer.fit_transform(df[numeric_features])
        df[categorical_features] = cat_imputer.fit_transform(df[categorical_features])

    # 4. Outlier Removal
    if outlier_removal_strategy:
        method = outlier_removal_strategy.get('method', 'z-score')
        threshold = outlier_removal_strategy.get('threshold', outlier_threshold)
        if method == 'z-score':
            for col in numeric_features:
                if df[col].nunique() > 1:  # Only process columns with variance
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    df = df[z_scores < threshold]
                else:
                    print(f"Warning: Column '{col}' has no variance, skipping outlier removal")

    # 5. Encoding
    encoded_df = df.copy()
    encoding_summary = {}
    for col in categorical_features:
        unique_values = df[col].nunique()
        if unique_values < max_encoded_columns and encoding_strategies.get('low_cardinality') == 'one-hot':
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
            encoded_data = encoder.fit_transform(df[[col]])
            encoded_column_names = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
            encoded_df = pd.concat([encoded_df, pd.DataFrame(encoded_data, columns=encoded_column_names, index=encoded_df.index)], axis=1)
            encoded_df.drop(columns=[col], inplace=True)
            encoding_summary[col] = f"One-hot encoded ({unique_values} categories)"
        elif encoding_strategies.get('high_cardinality') == 'ordinal':
            encoder = OrdinalEncoder()
            encoded_df[col] = encoder.fit_transform(df[[col]])
            encoding_summary[col] = f"Ordinal encoded ({unique_values} categories)"

    # 6. New Feature Engineering Summary
    feature_engineering_summary = {
        "imputation": imputation_strategies,
        "outlier_removal": outlier_removal_strategy,
        "encoding": encoding_summary,
        "new_features": new_features_suggestions
    }

    # 7. Feature Scaling
    scaler = StandardScaler()
    numerical_features_to_scale = [col for col in encoded_df.columns
                                    if encoded_df[col].dtype in [np.float64, np.int64] and col != target_column]
    if numerical_features_to_scale:
        encoded_df[numerical_features_to_scale] = scaler.fit_transform(encoded_df[numerical_features_to_scale])
        feature_engineering_summary["feature_scaling"] = "StandardScaler applied to numerical features (excluding target)."
    else:
        feature_engineering_summary["feature_scaling"] = "No numerical features to scale."

    return encoded_df, feature_engineering_summary
