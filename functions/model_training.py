from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, mean_squared_error, precision_score, 
                            recall_score, f1_score, r2_score, mean_absolute_error, 
                            explained_variance_score, roc_auc_score, log_loss)


from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, 
                             GradientBoostingClassifier, GradientBoostingRegressor, 
                             AdaBoostClassifier, AdaBoostRegressor, 
                             ExtraTreesClassifier, ExtraTreesRegressor)
from sklearn.linear_model import (LogisticRegression, LinearRegression, 
                               Ridge, Lasso, ElasticNet)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import re
import numpy as np

def select_target_column(df, llm):
    """Select the target column for modeling using LLM guidance.
    
    Args:
        df: Input DataFrame
        llm: Language model instance
        
    Returns:
        str: Selected target column name
        
    Raises:
        ValueError: If input validation fails or LLM response is invalid
    """

    """Model decides the target column.
    
    Args:
        df: Input DataFrame
        llm: Language model instance
        
    Returns:
        str: Selected target column name
        
    Raises:
        ValueError: If input validation fails or LLM response is invalid
    """
    # Input validation
    if df.empty:
        raise ValueError("DataFrame is empty. Cannot select target column.")
    
    if len(df.columns) == 0:
        raise ValueError("DataFrame has no columns. Cannot select target column.")
        
    if not hasattr(llm, 'invoke'):
        raise ValueError("Invalid LLM instance provided")

    # Pass the current columns of the dataframe to the prompt
    select_target_template = f"""Task: select_target_column
    Data: {df.head().to_json()}
    Columns: {', '.join(df.columns)}
    Instructions:
    - Carefully analyze the provided data and column names.
    - Identify the column that is most likely to be the target variable for prediction.
    - Consider the following factors:
        - The column's data type (numerical or categorical).
        - The column's name and its potential relationship to a prediction task.
        - The distribution of values in the column.
    - Return only one valid column name from the provided 'Columns' list, without any additional queries or SQL-like responses.
    - If you are uncertain, prioritize numerical columns and those with names suggestive of being a target variable (e.g., 'price', 'target', 'label').
    - If no suitable target column is found, return 'None'.
    """

    select_target_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a target column selection expert."),
        HumanMessage(content=select_target_template)
    ])
    target_decision = llm.invoke(select_target_prompt.format_messages(data=df.head().to_json(), columns=', '.join(df.columns)))

    if not hasattr(target_decision, 'content'):
        raise ValueError("Invalid response from LLM: Missing content attribute. Please check the LLM configuration.")

    # Clean up and extract the target column name
    target_column = target_decision.content.strip()
    
    # Validate the response format
    if not isinstance(target_column, str):
        raise ValueError(f"Invalid target column format. Expected string, got {type(target_column)}")

    # Handle case where LLM returns 'None'
    if target_column.lower() == 'none':
        print("Warning: LLM could not determine a suitable target column. Using the first column as fallback.")
        return df.columns[0]

    # Ensure the suggested target column exists in the dataframe
    if target_column not in df.columns:
        # Try to find a close match (case-insensitive)
        close_matches = [col for col in df.columns if col.lower() == target_column.lower()]
        if close_matches:
            target_column = close_matches[0]
            print(f"Info: Using case-insensitive match for target column: {target_column}")
        else:
            print(f"Warning: Model suggested an invalid target column: '{target_column}'. Using the first column ('{df.columns[0]}') as the target instead.")
            target_column = df.columns[0]  # Use the first column as a fallback
    
    # Validate the selected target column
    if df[target_column].nunique() == 1:
        print(f"Warning: Selected target column '{target_column}' has only one unique value. This may not be suitable for modeling.")

    return target_column


def model_selection_train_and_evaluation(df, target_col, llm, test_size=0.2, random_state=42, cv_folds=5):
    """Perform model selection, training, and evaluation with enhanced capabilities.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        llm: Language model instance
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        cv_folds: Number of cross-validation folds
        
    Returns:
        tuple: (best_model, model_type, X, y, summary, best_metrics)
        
    Raises:
        ValueError: If input validation fails
    """

    """Train and evaluate machine learning models.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        llm: Language model instance
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (best_model, model_type, X, y, summary, best_metrics)
        
    Raises:
        ValueError: If input validation fails
    """
    # Input validation
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    if df[target_col].isnull().all():
        raise ValueError(f"Target column '{target_col}' contains only null values")
    if not hasattr(llm, 'invoke'):
        raise ValueError("Invalid LLM instance provided")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Step 1: Model Type Selection
    model_type_template = f"""
    Task: model_selection
    Data: {df.head().to_json()}
    Target Column: {target_col}
    Target Column Data Type: {df[target_col].dtype}

    Instructions:
    - Determine whether the machine learning task is classification or regression.
    - Consider the following factors:
        - The data type of the target column ('{df[target_col].dtype}').
        - The distribution of values in the target column.
        - The overall goal is to predict the value of the target column based on other features.
    - If the target column is numerical and the goal is to predict a continuous value, choose 'regression'.
    - If the target column is categorical or the goal is to predict a class label, choose 'classification'.
    - If you are unable to confidently determine the task type based on the provided information, provide a brief explanation and suggest potential alternative approaches.

    Return only one of the following: 'classification', 'regression', or your alternative suggestion with an explanation.
    """

    model_type_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a model selection expert."),
        HumanMessage(content=model_type_template)
    ])

    model_type_decision = llm.invoke(model_type_prompt.format_messages(
        data=df.head().to_json(),
        target_column=target_col,
        target_column_dtype=df[target_col].dtype
    ))

    model_type = model_type_decision.content.strip().lower()

    if "classification" in model_type:
        model_type = "classification"
    elif "regression" in model_type:
        model_type = "regression"
    elif "other" in model_type:
        model_type = "other"
    else:
        raise ValueError(f"Invalid LLM response format. Could not determine model type from response: {model_type}")

    # Step 2: Define Models
    models = []
    metrics_dict = {}  # Store metrics for each model

    if model_type == "classification":
        models = [
            RandomForestClassifier(class_weight="balanced"),
            LogisticRegression(),
            KNeighborsClassifier(),
            DecisionTreeClassifier(),
            GradientBoostingClassifier(),
            AdaBoostClassifier(),
            GaussianNB(),
            XGBClassifier(eval_metric='logloss'),
        ]
        metrics_dict = {model.__class__.__name__: {"accuracy": None, "precision": None, "recall": None, "f1_score": None} for model in models}
    elif model_type == "regression":
        models = [
            RandomForestRegressor(),
            LinearRegression(),
            DecisionTreeRegressor(),
            GradientBoostingRegressor(),
            AdaBoostRegressor(),
            XGBRegressor(eval_metric='logloss'),
        ]
        metrics_dict = {model.__class__.__name__: {"mean_squared_error": None} for model in models}
    else:
        raise ValueError(f"Unsupported model type: {model_type}.")

    # Step 3: Train Test Split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y if model_type == "classification" else None
        )
    except ValueError as e:
        raise ValueError(f"Error during train-test split: {str(e)}. Please check your data distribution.")

    # Step 4: Hyperparameter Tuning, Model Training, and Evaluation
    best_model = None
    best_score = float('-inf')  # For classification, it's accuracy, for regression, it's MSE
    best_metrics = {}
    best_hyperparameters = {}  # Store best hyperparameters
    models_used = [model_instance.__class__.__name__ for model_instance in models]

    # Imputation (using SimpleImputer for demonstration)
    imputer = SimpleImputer(strategy='most_frequent')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    for model_instance in models:
        model_name = model_instance.__class__.__name__
        print(f"Evaluating model: {model_name}")

        try:
            # Handle imbalanced datasets for classification
            if model_type == "classification":
                unique_classes, class_counts = np.unique(y_train, return_counts=True)
                if len(unique_classes) > 1 and min(class_counts) / max(class_counts) < 0.2:  # Imbalance threshold
                    print("Applying SMOTE to handle class imbalance...")
                    smote = SMOTE(random_state=42)
                    X_train_imputed, y_train = smote.fit_resample(X_train_imputed, y_train)  # Use imputed data for SMOTE

            # Hyperparameter Tuning using GridSearchCV
            param_grid = {}
            if model_name == "RandomForestClassifier":
                param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5, 10]
            }
            elif model_name == "LogisticRegression":
                param_grid = {
                    "C": [0.1, 1, 10],
                    "penalty": ['l1', 'l2']
                }
            grid_search = GridSearchCV(model_instance, param_grid, cv=5, scoring='accuracy' if model_type == "classification" else 'neg_mean_squared_error')
            grid_search.fit(X_train_imputed, y_train)  # Use imputed data for GridSearchCV

            # Update model_instance with the best estimator
            model_instance = grid_search.best_estimator_
            best_hyperparameters[model_name] = grid_search.best_params_

            # Model Fitting and Prediction
            model_instance.fit(X_train_imputed, y_train)
            y_pred = model_instance.predict(X_test_imputed)

            # Metrics Calculation
            if model_type == "classification":
                score = accuracy_score(y_test, y_pred)
                metrics = {
                    "accuracy": score,
                    "precision": precision_score(y_test, y_pred, average='weighted'),
                    "recall": recall_score(y_test, y_pred, average='weighted'),
                    "f1_score": f1_score(y_test, y_pred, average='weighted')
                }
                print(f"{metrics}")
            elif model_type == "regression":
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                evs = explained_variance_score(y_test, y_pred)
                
                metrics = {
                    "mean_squared_error": mse,
                    "root_mean_squared_error": rmse,
                    "mean_absolute_error": mae,
                    "r2_score": r2,
                    "explained_variance_score": evs
                }
                print(f"{metrics}")
                # Use negative MSE for scoring to maintain consistency with GridSearchCV
                score = -mse


            # Store metrics for the current model
            metrics_dict[model_name] = metrics

            # Store the best model
            if score > best_score:

                best_model = model_instance
                best_score = score
                best_metrics = metrics

        except Exception as e:
            print(f"Error evaluating model {model_name}: {e}")
            continue

    # Create a summary dictionary
    summary = {
        "model_type": model_type,
        "models_evaluated": models_used,  # List of models evaluated
        "model_metrics": metrics_dict,  # Metrics for each model
        "best_model": best_model.__class__.__name__,  # Name of the best model
        "best_model_metrics": best_metrics,  # Metrics of the best model
        "best_hyperparameters": best_hyperparameters  # Store best hyperparameters
    }

    # Structure best_metrics with model name, metrics, and parameters
    structured_best_metrics = {
        "model_name": best_model.__class__.__name__,
        "metrics": best_metrics,
        "parameters": best_hyperparameters.get(best_model.__class__.__name__, {})


    }
    return best_model, model_type, X, y, summary, structured_best_metrics
