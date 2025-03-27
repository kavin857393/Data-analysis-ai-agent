from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

import streamlit as st
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,SimpleImputer

from sklearn.impute import IterativeImputer,SimpleImputer
def remove_unwanted_columns(df,llm):
    """Agent decides which columns should be removed, with human approval and summary."""
    remove_unwanted_columns_template = f"""
Task: remove_unwanted_columns
Data: {df.head().to_json()}
Instructions:
1. Analyze the given data and identify columns that are irrelevant or potentially harmful for the model.
2. Consider factors like:
    - High percentage of missing values
    - Low variance or constant values
    - Redundant information or highly correlated features
    - Features that are unlikely to be predictive
    - Features that could introduce bias or fairness issues
3. Return a list of columns to remove along with a brief explanation for each removal.

Example Response:
- **'column_name_1'**: Removed due to high percentage of missing values.
- **'column_name_2'**: Removed due to low variance.
- **'column_name_3'**: Removed due to redundancy with 'column_name_4'.
    """
    
    remove_unwanted_columns_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(content=remove_unwanted_columns_template)
    ])

    # Invoke the LLM with the prompt (assuming llm.invoke returns an object with a 'content' attribute)
    decision = llm.invoke(remove_unwanted_columns_prompt.format_messages(data=df.head().to_json()))

    if not hasattr(decision, 'content'):
        raise ValueError("Invalid response from llm.")

    # Parse the LLM's response to extract columns and explanations
    columns_to_remove = []
    removal_explanations = {}  # Store explanations for each column

    for line in decision.content.split('\n'):
        line = line.strip()
        if line.startswith('-'):
            parts = line[2:].split(':')  # Split by colon to separate column and reason
            if len(parts) >= 2:
                column_name = parts[0].strip().strip("'\"")  # Remove quotes and whitespace
                explanation = parts[1].strip()

                if column_name in df.columns:
                    columns_to_remove.append(column_name)
                    removal_explanations[column_name] = explanation

    # Present the agent's decision to the human using Streamlit UI elements.
    st.write("### Agent's Suggested Column Removals")
    for col, reason in removal_explanations.items():
        st.write(f"- **{col}**: {reason}")

    user_approval = st.radio("Do you want to remove these columns?", ("Yes", "No"))

    if user_approval.lower() == "yes":
        df_cleaned = df.drop(columns=columns_to_remove)
    else:
        # Only proceed with removal if user explicitly provides columns
        user_input_columns = st.text_input("Enter the columns to remove, separated by commas (or leave blank to keep all columns):")
        if user_input_columns.strip():
            manual_columns = [col.strip() for col in user_input_columns.split(',')]
            # Ensure the manually specified columns exist in the dataframe
            columns_to_remove = [col for col in manual_columns if col in df.columns]
            if columns_to_remove:
                df_cleaned = df.drop(columns=columns_to_remove)
                # Update removal explanations for manually specified columns
                for col in columns_to_remove:
                    removal_explanations[col] = "Removed based on user input."
            else:
                st.warning("None of the specified columns exist in the dataset.")
                df_cleaned = df.copy()
        else:
            st.info("No columns were removed - keeping all columns.")
            df_cleaned = df.copy()
            columns_to_remove = []
            removal_explanations = {}
        
        # Clear AI-suggested columns if user rejected them
        columns_to_remove = []
        removal_explanations = {}



    # Create a summary of removed columns and their explanations
    remove_summary = {
        "removed_columns": columns_to_remove,
        "removal_explanations": removal_explanations,
    }

    return df_cleaned, remove_summary, columns_to_remove


def data_cleaning(df, llm, missing_threshold=0.8):
    """Handles missing values, duplicates, and basic cleanup using prompts.
    
    Args:
        df: Input DataFrame
        llm: Language model instance
        missing_threshold: Threshold for removing columns with high missing values
        
    Returns:
        tuple: (cleaned_df, summary_dict)
    """


    # Construct the prompt for the LLM
    cleaning_template = f"""
    Task: data_cleaning
    Data: {df.head().to_json()}
    Instructions:
    1. Analyze the provided data and identify potential data quality issues, such as missing values, duplicates, and inconsistencies.
    2. Recommend specific cleaning steps to address these issues, including:
        - Handling missing values using appropriate methods:
            * For numerical data: mean, median, mode, KNN imputation, or iterative imputation
            * For categorical data: most frequent value or create new category
        - Removing duplicate rows.
        - Correcting inconsistencies in data formats (if any).
        - Handling outliers using appropriate methods.
    3. Provide a concise summary of the recommended cleaning steps.

    Example Response:
    Recommended Cleaning Steps:
    - Remove columns with more than 80% missing values.
    - For numerical features:
        * Use KNN imputation for features with moderate missingness
        * Use iterative imputation for features with complex relationships
        * Use median imputation for features with outliers
    - For categorical features:
        * Use most frequent value for features with low cardinality
        * Create 'missing' category for features with high cardinality
    - Remove duplicate rows.
    - Handle outliers using z-score method with threshold of 3.
    """


    # Get cleaning suggestions from the LLM
    cleaning_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a data cleaning expert with knowledge of advanced imputation techniques."),
        HumanMessage(content=cleaning_template)
    ])
    cleaning_decision = llm.invoke(cleaning_prompt.format_messages(data=df.head().to_json()))



    if not hasattr(cleaning_decision, 'content'):
        raise ValueError("Invalid response from llm.")

    df_cleaned = df.copy()
    summary = {
        "actions": [],
        "imputation_methods": {},
        "outlier_handling": {}
    }  # Initialize summary


    # --- Parsing and Applying Cleaning Steps ---
    for line in cleaning_decision.content.split('\n'):
        line = line.strip().lower()

        # Removing columns with high missing values
        if "remove columns with more than" in line:
            try:
                threshold_percentage = float(line.split("remove columns with more than")[1].split("%")[0].strip())
                threshold = threshold_percentage / 100.0
                high_missing_columns = df_cleaned.columns[df_cleaned.isnull().mean() > threshold]
                df_cleaned = df_cleaned.drop(columns=high_missing_columns)
                summary["actions"].append(f"Removed columns with more than {threshold_percentage}% missing values: {high_missing_columns.tolist()}")
            except ValueError:
                print("Warning: Could not parse missing value threshold. Skipping this step.")

        # Imputing missing numerical values
        elif "impute missing numerical values with" in line:
            strategy = line.split("impute missing numerical values with")[1].strip()
            numeric_features = df_cleaned.select_dtypes(include=np.number).columns
            
            if "knn" in strategy:
                imputer = KNNImputer(n_neighbors=5)
                df_cleaned[numeric_features] = imputer.fit_transform(df_cleaned[numeric_features])
                summary["imputation_methods"]["numerical"] = "KNN imputation"
                summary["actions"].append("Used KNN imputation for numerical features")
            elif "iterative" in strategy:
                imputer = IterativeImputer(max_iter=10, random_state=42)
                df_cleaned[numeric_features] = imputer.fit_transform(df_cleaned[numeric_features])
                summary["imputation_methods"]["numerical"] = "Iterative imputation"
                summary["actions"].append("Used iterative imputation for numerical features")
            elif strategy in ["mean", "median", "most_frequent"]:
                imputer = SimpleImputer(strategy=strategy)
                df_cleaned[numeric_features] = imputer.fit_transform(df_cleaned[numeric_features])
                summary["imputation_methods"]["numerical"] = f"{strategy} imputation"
                summary["actions"].append(f"Imputed missing numerical values with the {strategy}")
            else:
                print(f"Warning: Unknown imputation strategy '{strategy}' for numerical values. Using median as fallback.")
                imputer = SimpleImputer(strategy="median")
                df_cleaned[numeric_features] = imputer.fit_transform(df_cleaned[numeric_features])
                summary["imputation_methods"]["numerical"] = "median imputation (fallback)"
                summary["actions"].append("Used median imputation as fallback for numerical features")

        # Imputing missing categorical values
        elif "impute missing categorical values with" in line:
            strategy = line.split("impute missing categorical values with")[1].strip()
            categorical_features = df_cleaned.select_dtypes(exclude=np.number).columns
            
            if "create missing category" in strategy:
                for col in categorical_features:
                    df_cleaned[col] = df_cleaned[col].fillna("missing")
                summary["imputation_methods"]["categorical"] = "created 'missing' category"
                summary["actions"].append("Created 'missing' category for categorical features")
            elif strategy in ["most_frequent"]:
                imputer = SimpleImputer(strategy=strategy)
                df_cleaned[categorical_features] = imputer.fit_transform(df_cleaned[categorical_features])
                summary["imputation_methods"]["categorical"] = "most frequent value"
                summary["actions"].append(f"Imputed missing categorical values with the {strategy}")
            else:
                print(f"Warning: Unknown imputation strategy '{strategy}' for categorical values. Using most frequent as fallback.")
                imputer = SimpleImputer(strategy="most_frequent")
                df_cleaned[categorical_features] = imputer.fit_transform(df_cleaned[categorical_features])
                summary["imputation_methods"]["categorical"] = "most frequent value (fallback)"
                summary["actions"].append("Used most frequent value as fallback for categorical features")

        # Removing duplicate rows
        elif "remove duplicate rows" in line:
            num_duplicates = df_cleaned.duplicated().sum()
            df_cleaned = df_cleaned.drop_duplicates()
            summary["actions"].append(f"Removed {num_duplicates} duplicate rows")

        # Handling outliers
        elif "handle outliers using" in line:
            method = line.split("handle outliers using")[1].strip()
            if "z-score" in method:
                threshold = 3  # Default threshold
                if "threshold of" in line:
                    try:
                        threshold = float(line.split("threshold of")[1].strip())
                    except ValueError:
                        print("Warning: Could not parse outlier threshold. Using default value of 3.")
                
                for col in df_cleaned.select_dtypes(include=np.number).columns:
                    z_scores = np.abs((df_cleaned[col] - df_cleaned[col].mean()) / df_cleaned[col].std())
                    df_cleaned = df_cleaned[z_scores < threshold]
                summary["outlier_handling"] = {
                    "method": "z-score",
                    "threshold": threshold
                }
                summary["actions"].append(f"Removed outliers using z-score method with threshold {threshold}")
    # --- End of Parsing and Applying Cleaning Steps ---


    return df_cleaned, summary
