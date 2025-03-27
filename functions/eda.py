from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import io
import os

import numpy as np
import pandas as pd
import matplotlib
# Switch to a non-interactive backend to avoid interactive warnings.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import re
from io import StringIO
import os
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def perform_eda(df, llm):
    """Perform comprehensive exploratory data analysis.
    
    Args:
        df: Input DataFrame
        llm: Language model instance for guidance
        
    Returns:
        dict: Contains summary, visualizations, and main visualization path
    """
    # Create output directory if it doesn't exist
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "files_and_models", "eda_output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving EDA visualizations to: {output_dir}")


    # Initialize results
    summary = ""
    visualization_paths = []
    statistical_tests = {}
    
    # Initialize main visualization path with default correlation matrix
    main_visualization_path = os.path.join(output_dir, "correlation_matrix.png")



    eda_template = """
    Task: Comprehensive Exploratory Data Analysis (EDA)


    Data: {df.head().to_json()}
    Data Description: The data contains information about [briefly describe the data, e.g., customer demographics, product sales, financial transactions].
    Shape: {df.shape}
    Columns: {', '.join(df.columns)}


    Instructions:
    1. Perform comprehensive analysis of the provided data and identify key insights and patterns.
    2. Generate Python code using 'seaborn', 'matplotlib.pyplot', and statistical libraries to create visualizations and statistical tests.
    3. Consider the following analyses:
        - Descriptive statistics and data distribution
        - Correlation and covariance analysis
        - Feature importance and mutual information
        - Statistical significance tests
    4. Create the following visualizations:

        - Histograms/Distributions: For numerical features, to show the distribution of values.
        - Box Plots: For numerical features, to show the distribution and identify outliers.
        - Count Plots/Bar Charts: For categorical features, to show the frequency of each category.
        - Scatter Plots: For pairs of numerical features, to show the relationship between them.
        - Correlation Matrix Heatmap: To show the correlation between numerical features.
        - Violin Plots: To show the distribution of numerical data across different categories.
        - Line Plots: For time-series data, if applicable.
        - Pair Plots: For multivariate analysis of numerical features.
        - Feature Importance Plots: To show the importance of features using mutual information.
    5. Perform statistical tests:
        - Normality tests (Shapiro-Wilk, Kolmogorov-Smirnov)
        - Correlation significance tests
        - ANOVA or Chi-square tests for categorical variables
    6. For each analysis and visualization, provide a brief textual summary of the insights gained.

    """
    eda_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful AI assistant for data analysis."),
        HumanMessage(content=eda_template)
    ])

    # Wrap literal JSON in StringIO to avoid deprecation warnings.
    data_json = df.head().to_json()
    eda_response = llm.invoke(eda_prompt.format_messages(data=data_json))



    # Initialize results
    summary = ""
    visualization_paths = []

    # --- EDA Tasks and Summary Generation ---
    if "describe" in eda_response.content.lower():
        summary += "## Descriptive Statistics:\n\n"
        summary += df.describe(include='all').to_markdown() + "\n\n"


    if "info" in eda_response.content.lower():
        summary += "## Data Information:\n\n"
        buffer = StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        summary += info_str + "\n\n"

    if any(keyword in eda_response.content.lower() for keyword in ["missing values", "null values", "nan values"]):
        summary += "## Missing Values:\n\n"
        missing_values = df.isnull().sum()
        summary += missing_values.to_frame('Missing Values').to_markdown() + "\n\n"


    if any(keyword in eda_response.content.lower() for keyword in ["data types", "column types"]):
        summary += "## Data Types:\n\n"
        data_types = df.dtypes
        summary += data_types.to_frame('Data Type').to_markdown() + "\n\n"


    # --- Visualization Logic ---
    # Histogram - Create main visualization
    if "histogram" in eda_response.content.lower():
        numerical_cols = df.select_dtypes(include=np.number).columns
        if len(numerical_cols) > 0:
            # Use first numerical column for main visualization
            main_col = numerical_cols[0]
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df[main_col], kde=True, ax=ax)
            ax.set_title(f"Distribution of {main_col}")
            main_visualization_path = os.path.abspath(os.path.join(output_dir, "eda_output.png"))
            fig.savefig(main_visualization_path)
            visualization_paths.append(main_visualization_path)
            
            # Create histograms for other numerical columns
            for col in numerical_cols[1:]:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.histplot(df[col], kde=True, ax=ax)
                ax.set_title(f"Distribution of {col}")
                file_path = os.path.join(output_dir, f"histogram_{col}.png")
                fig.savefig(file_path)
                visualization_paths.append(file_path)

    # Box Plot (with Outlier Detection)
    if "box plot" in eda_response.content.lower():
        for col in df.select_dtypes(include=np.number).columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(y=df[col], ax=ax)
            ax.set_title(f"Box Plot of {col} (with Outliers)")
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5 * IQR
            lower_bound = Q1 - 1.5 * IQR
            outliers = df[(df[col] > upper_bound) | (df[col] < lower_bound)]
            ax.plot(outliers.index, outliers[col], 'ro', markersize=5)
            file_path = os.path.join(output_dir, f"boxplot_{col}.png")
            fig.savefig(file_path)
            visualization_paths.append(file_path)

    # Count Plot / Bar Plot
    if any(keyword in eda_response.content.lower() for keyword in ["count plot", "bar plot", "bar chart"]):
        for col in df.select_dtypes(include=['object', 'category']).columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(x=df[col], ax=ax)
            ax.set_title(f"Count Plot of {col}")
            plt.xticks(rotation=45, ha='right')
            file_path = os.path.join(output_dir, f"countplot_{col}.png")
            fig.savefig(file_path)
            visualization_paths.append(file_path)

    # Scatter Plot
    if "scatter plot" in eda_response.content.lower():
        numerical_cols = list(df.select_dtypes(include=np.number).columns)
        for i in range(len(numerical_cols)):
            for j in range(i + 1, len(numerical_cols)):
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.scatterplot(x=df[numerical_cols[i]], y=df[numerical_cols[j]], ax=ax)
                ax.set_title(f"Scatter Plot of {numerical_cols[i]} vs {numerical_cols[j]}")
                file_path = os.path.join(output_dir, f"scatter_{numerical_cols[i]}_vs_{numerical_cols[j]}.png")
                fig.savefig(file_path)
                visualization_paths.append(file_path)

    # Always create correlation matrix as default visualization
    numerical_features = list(df.select_dtypes(include=np.number).columns)
    if len(numerical_features) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[numerical_features].corr(), annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Matrix")
        file_path = os.path.join(output_dir, "correlation_matrix.png")
        fig.savefig(file_path)
        visualization_paths.append(file_path)
        main_visualization_path = file_path


    # Violin Plot
    if "violin plot" in eda_response.content.lower():
        for col in df.select_dtypes(include=np.number).columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.violinplot(y=df[col], ax=ax)
            ax.set_title(f"Violin Plot of {col}")
            file_path = os.path.join(output_dir, f"violinplot_{col}.png")
            fig.savefig(file_path)
            visualization_paths.append(file_path)

    # Line Plot
    if "line plot" in eda_response.content.lower():
        fig, ax = plt.subplots(figsize=(10, 6))
        if 'time_column' in df.columns and 'value_column' in df.columns:
            sns.lineplot(x='time_column', y='value_column', data=df, ax=ax)
            ax.set_title("Line Plot")
            file_path = os.path.join(output_dir, "lineplot.png")
            fig.savefig(file_path)
            visualization_paths.append(file_path)

    # Pair Plot
    if "pair plot" in eda_response.content.lower():
        pairplot_fig = sns.pairplot(df)
        file_path = os.path.join(output_dir, "pairplot.png")
        pairplot_fig.savefig(file_path)
        visualization_paths.append(file_path)

    # Joint Plot
    if "joint plot" in eda_response.content.lower():
        x_col = "numerical_column_1"
        y_col = "numerical_column_2"
        if x_col in df.columns and y_col in df.columns:
            jointplot_fig = sns.jointplot(x=df[x_col], y=df[y_col])
            file_path = os.path.join(output_dir, "jointplot.png")
            jointplot_fig.savefig(file_path)
            visualization_paths.append(file_path)

    # KDE Plot
    if "kde plot" in eda_response.content.lower():
        for col in df.select_dtypes(include=np.number).columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.kdeplot(df[col], ax=ax)
            ax.set_title(f"KDE Plot of {col}")
            file_path = os.path.join(output_dir, f"kdeplot_{col}.png")
            fig.savefig(file_path)
            visualization_paths.append(file_path)

    # Rug Plot
    if "rug plot" in eda_response.content.lower():
        for col in df.select_dtypes(include=np.number).columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.rugplot(df[col], ax=ax)
            ax.set_title(f"Rug Plot of {col}")
            file_path = os.path.join(output_dir, f"rugplot_{col}.png")
            fig.savefig(file_path)
            visualization_paths.append(file_path)

    # Strip Plot
    if "strip plot" in eda_response.content.lower():
        for col in df.select_dtypes(include=['object', 'category']).columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.stripplot(x=df[col], ax=ax)
            ax.set_title(f"Strip Plot of {col}")
            plt.xticks(rotation=45, ha='right')
            file_path = os.path.join(output_dir, f"stripplot_{col}.png")
            fig.savefig(file_path)
            visualization_paths.append(file_path)

    # Swarm Plot
    if "swarm plot" in eda_response.content.lower():
        for col in df.select_dtypes(include=['object', 'category']).columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.swarmplot(x=df[col], ax=ax)
            ax.set_title(f"Swarm Plot of {col}")
            plt.xticks(rotation=45, ha='right')
            file_path = os.path.join(output_dir, f"swarmplot_{col}.png")
            fig.savefig(file_path)
            visualization_paths.append(file_path)

    # Boxen Plot
    if "boxen plot" in eda_response.content.lower():
        for col in df.select_dtypes(include=np.number).columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxenplot(y=df[col], ax=ax)
            ax.set_title(f"Boxen Plot of {col}")
            file_path = os.path.join(output_dir, f"boxenplot_{col}.png")
            fig.savefig(file_path)
            visualization_paths.append(file_path)

    # Point Plot
    if "point plot" in eda_response.content.lower():
        x_col = "categorical_column"
        y_col = "numerical_column"
        if x_col in df.columns and y_col in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.pointplot(x=x_col, y=y_col, data=df, ax=ax)
            ax.set_title("Point Plot")
            plt.xticks(rotation=45, ha='right')
            file_path = os.path.join(output_dir, "pointplot.png")
            fig.savefig(file_path)
            visualization_paths.append(file_path)

    # Additional plots: Andrews Curves, Parallel Coordinates, Radviz
    if "andrews curves" in eda_response.content.lower():
        from pandas.plotting import andrews_curves
        fig, ax = plt.subplots(figsize=(12, 6))
        target_column = "target_column"
        if target_column in df.columns:
            andrews_curves(df, target_column, ax=ax)
            ax.set_title("Andrews Curves")
            file_path = os.path.join(output_dir, "andrews_curves.png")
            fig.savefig(file_path)
            visualization_paths.append(file_path)

    if "parallel coordinates" in eda_response.content.lower():
        from pandas.plotting import parallel_coordinates
        fig, ax = plt.subplots(figsize=(12, 6))
        target_column = "target_column"
        if target_column in df.columns:
            parallel_coordinates(df, target_column, ax=ax)
            ax.set_title("Parallel Coordinates")
            file_path = os.path.join(output_dir, "parallel_coordinates.png")
            fig.savefig(file_path)
            visualization_paths.append(file_path)

    if "radviz" in eda_response.content.lower():
        from pandas.plotting import radviz
        fig, ax = plt.subplots(figsize=(8, 8))
        target_column = "target_column"
        if target_column in df.columns:
            radviz(df, target_column, ax=ax)
            ax.set_title("Radviz")
            file_path = os.path.join(output_dir, "radviz.png")
            fig.savefig(file_path)
            visualization_paths.append(file_path)

    # Execute any additional plotting code from the LLM response
    code_blocks = re.findall(r"```python(.*?)```", eda_response.content, re.DOTALL)
    for block in code_blocks:
        try:
            local_vars = {'df': df, 'np': np, 'plt': plt, 'sns': sns}
            exec(block.strip(), local_vars)
            file_path = os.path.join(output_dir, "llm_plot.png")
            plt.savefig(file_path)
            visualization_paths.append(file_path)
        except Exception as e:
            pass

    final_summary =  "\n\n" + summary
    
    # Create pair plot if no other visualizations were created
    if len(visualization_paths) == 0:
        pairplot_fig = sns.pairplot(df)
        file_path = os.path.join(output_dir, "pairplot.png")
        pairplot_fig.savefig(file_path)
        visualization_paths.append(file_path)
        main_visualization_path = file_path

    
    return {
        'summary': final_summary,
        'visualizations': visualization_paths,
        'main_visualization': main_visualization_path
    }
