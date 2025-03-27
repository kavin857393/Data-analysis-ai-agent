import streamlit as st
import time
import pandas as pd
import os
import joblib
from PIL import Image
from langgraph.graph import StateGraph
from langchain.schema.runnable import RunnableLambda
from typing import Dict, Any, TypedDict, Optional
import logging

# Import custom modules (ensure these exist in your project)
from database.db import init_db
from functions.data_cleaning import remove_unwanted_columns, data_cleaning
from functions.eda import perform_eda
from functions.feature_engineering import feature_engineering
from functions.model_training import model_selection_train_and_evaluation, select_target_column
from functions.storage import store_summary_in_db
from workflows.rag import answer_queries_using_rag
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from config import settings

# ----------------------------- Configuration & Logging -----------------------------
st.set_page_config(page_title="ML Agent UI", layout="wide", initial_sidebar_state="expanded")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure database is initialized
if not init_db():
    st.error("Failed to initialize database")
    st.stop()

# ----------------------------- Session State Initialization -----------------------------
def init_session_state():
    keys = [
        "chat_histories", "current_file", "df", "df_after_removal", "remove_summary",
        "columns_removed", "df_cleaned", "cleaning_summary", "eda_summary",
        "eda_results", "expand_data_cleaning", "expand_target_selection", "target_selected",
        "df_engineered", "feature_summary", "best_model", "summary", "best_metrics",
        "training_completed", "workflow_progress", "last_retrieved_context", "rag_status"
    ]
    for key in keys:
        if key not in st.session_state:
            st.session_state[key] = None

    if st.session_state["chat_histories"] is None:
        st.session_state["chat_histories"] = {}
    if st.session_state["current_file"] is None:
        st.session_state["current_file"] = "default_chat"
    if st.session_state["workflow_progress"] is None:
        st.session_state["workflow_progress"] = {}

init_session_state()

# ----------------------------- Sidebar: Configuration & Reset -----------------------------
with st.sidebar:
    st.header("Configuration")
    selected_model = st.selectbox(
        "Select LLM Model",
        [
            "llama-3.3-70b-versatile",
            "deepseek-r1-distill-llama-70b",
            "deepseek-r1-distill-qwen-32b",
            "gemma2-9b-it",
            "mixtral-8x7b-32768"
        ]
    )
    st.info("Choose an LLM model for your ML workflow analysis.")

    st.write("---")
    st.header("Instructions")
    st.markdown(""" 
    **Workflow Steps:**
    1. Upload CSV  
    2. Column Removal  
    3. Data Cleaning  
    4. EDA  
    5. Target Selection  
    6. Feature Engineering  
    7. Model Training  
    8. Store Summaries  
    9. Downloads  
    10. Chatbot  
    """)

    if st.button("Reset Workflow"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

# ----------------------------- Initialize LLM -----------------------------
llm = ChatGroq(model=selected_model, api_key=settings.groq_api_key)

# ----------------------------- Define Workflow State Schema -----------------------------
class WorkflowState(TypedDict):
    df: pd.DataFrame
    summaries: Dict[str, Any]
    target_column: str
    model: Any
    metrics: Dict[str, float]
    current_step: str
    file_name: str
    chat_histories: Dict[str, Any]
    current_file: str
    error: Optional[str]
    progress: float  # 0.0 to 1.0
    last_successful_step: str

# ----------------------------- Workflow Node Functions -----------------------------
def load_data(state: WorkflowState) -> WorkflowState:
    try:
        if "df" not in st.session_state or st.session_state.df is None:
            raise ValueError("No data loaded")
        if st.session_state.df.empty:
            raise ValueError("Uploaded file is empty")
        state["df"] = st.session_state.df
        state["summaries"] = {}
        state["current_step"] = "load_data"
        state["progress"] = 0.1
        state["last_successful_step"] = "load_data"
        state["error"] = None
        logger.info("Data loaded successfully")
    except Exception as e:
        state["error"] = str(e)
        st.error(f"Error loading data: {e}")
        logger.error(f"Error loading data: {e}")
    return state

def remove_columns(state: WorkflowState) -> WorkflowState:
    try:
        if st.session_state.df_after_removal is None:
            raise ValueError("Column removal step not completed")
        state["df"] = st.session_state.df_after_removal
        state["summaries"]["column_removal"] = st.session_state.remove_summary
        state["current_step"] = "remove_columns"
        state["progress"] = 0.2
        state["last_successful_step"] = "remove_columns"
        logger.info("Columns removed successfully")
    except Exception as e:
        state["error"] = str(e)
        st.error(f"Error in remove_columns: {e}")
        logger.error(f"Error in remove_columns: {e}")
    return state

def clean_data(state: WorkflowState) -> WorkflowState:
    try:
        if st.session_state.df_cleaned is None:
            raise ValueError("Data cleaning step not completed")
        state["df"] = st.session_state.df_cleaned
        state["summaries"]["data_cleaning"] = st.session_state.cleaning_summary
        state["current_step"] = "clean_data"
        state["progress"] = 0.3
        state["last_successful_step"] = "clean_data"
        logger.info("Data cleaned successfully")
    except Exception as e:
        state["error"] = str(e)
        st.error(f"Error in clean_data: {e}")
        logger.error(f"Error in clean_data: {e}")
    return state

def data_eda(state: WorkflowState) -> WorkflowState:
    try:
        if st.session_state.eda_summary is None:
            raise ValueError("EDA step not completed")
        state["summaries"]["eda"] = st.session_state.eda_summary
        state["current_step"] = "data_eda"
        state["progress"] = 0.4
        state["last_successful_step"] = "data_eda"
        logger.info("EDA completed successfully")
    except Exception as e:
        state["error"] = str(e)
        st.error(f"Error in data_eda: {e}")
        logger.error(f"Error in data_eda: {e}")
    return state

def select_target(state: WorkflowState) -> WorkflowState:
    try:
        if st.session_state.target_selected is None:
            raise ValueError("Target column selection not completed")
        state["target_column"] = st.session_state.target_selected
        state["current_step"] = "select_target"
        state["progress"] = 0.5
        state["last_successful_step"] = "select_target"
        logger.info("Target column selected")
    except Exception as e:
        state["error"] = str(e)
        st.error(f"Error in select_target: {e}")
        logger.error(f"Error in select_target: {e}")
    return state

def engineer_features(state: WorkflowState) -> WorkflowState:
    try:
        if st.session_state.df_engineered is None:
            raise ValueError("Feature engineering not completed")
        state["df"] = st.session_state.df_engineered
        state["summaries"]["feature_engineering"] = st.session_state.feature_summary
        state["current_step"] = "feature_engineering"
        state["progress"] = 0.6
        state["last_successful_step"] = "feature_engineering"
        logger.info("Feature engineering completed")
    except Exception as e:
        state["error"] = str(e)
        st.error(f"Error in engineer_features: {e}")
        logger.error(f"Error in engineer_features: {e}")
    return state

def train_model(state: WorkflowState) -> WorkflowState:
    try:
        if st.session_state.best_model is None:
            raise ValueError("Model training not completed")
        state["model"] = st.session_state.best_model
        state["metrics"] = st.session_state.best_metrics
        state["summaries"]["model_training"] = st.session_state.summary
        state["current_step"] = "train_model"
        state["progress"] = 0.8
        state["last_successful_step"] = "train_model"
        logger.info("Model training completed")
    except Exception as e:
        state["error"] = str(e)
        st.error(f"Error in train_model: {e}")
        logger.error(f"Error in train_model: {e}")
    return state

def store_summaries(state: WorkflowState) -> WorkflowState:
    try:
        # Ensure all summaries are present
        required = ["summary", "remove_summary", "cleaning_summary", "eda_summary", "feature_summary", "best_metrics", "file_name"]
        if not all(getattr(st.session_state, key, None) is not None for key in required):
            raise ValueError("Not all summaries are available")
        store_summary_in_db(
            summary=st.session_state.summary,
            remove_summary=st.session_state.remove_summary,
            cleaning_summary=st.session_state.cleaning_summary,
            eda_summary=st.session_state.eda_summary,
            feature_summary=st.session_state.feature_summary,
            best_metrics=st.session_state.best_metrics,
            file_name=st.session_state.file_name
        )
        state["current_step"] = "store_summaries"
        state["progress"] = 1.0
        state["last_successful_step"] = "store_summaries"
        logger.info("Summaries stored in DB successfully")
    except Exception as e:
        state["error"] = str(e)
        st.error(f"Error in store_summaries: {e}")
        logger.error(f"Error in store_summaries: {e}")
    return state

def log_step(state: WorkflowState) -> WorkflowState:
    logger.info(f"Step: {state['current_step']} | Progress: {state['progress']*100:.1f}%")
    if state.get("error"):
        logger.error(f"Error in step {state['current_step']}: {state['error']}")
    return state

# ----------------------------- Build and Compile Workflow -----------------------------
workflow = StateGraph(WorkflowState)
workflow.add_node("load_data", RunnableLambda(load_data))
workflow.add_node("remove_columns", RunnableLambda(remove_columns))
workflow.add_node("clean_data", RunnableLambda(clean_data))
workflow.add_node("data_eda", RunnableLambda(data_eda))
workflow.add_node("select_target", RunnableLambda(select_target))
workflow.add_node("engineer_features", RunnableLambda(engineer_features))
workflow.add_node("train_model", RunnableLambda(train_model))
workflow.add_node("store_summaries", RunnableLambda(store_summaries))
workflow.add_node("log_step", RunnableLambda(log_step))

workflow.set_entry_point("load_data")
workflow.add_edge("load_data", "remove_columns")
workflow.add_edge("remove_columns", "clean_data")
workflow.add_edge("clean_data", "data_eda")
workflow.add_edge("data_eda", "select_target")
workflow.add_edge("select_target", "engineer_features")
workflow.add_edge("engineer_features", "train_model")
workflow.add_edge("train_model", "store_summaries")
workflow.add_edge("store_summaries", "log_step")

app = workflow.compile()

# ----------------------------- Main Title -----------------------------
st.title("Welcome to Machine Learning Agent")

# ----------------------------- Step 1: File Upload -----------------------------
with st.expander("Step 1: Upload CSV File", expanded=True):
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            st.dataframe(df.head(), height=200)
            st.session_state.df = df
            st.session_state.file_name = uploaded_file.name  # Save file name
            st.session_state.current_file = uploaded_file.name
            if st.session_state.current_file not in st.session_state.chat_histories:
                st.session_state.chat_histories[st.session_state.current_file] = []
        except Exception as e:
            st.error(f"Error reading file: {e}")

# ----------------------------- Step 2: Column Removal -----------------------------
if st.session_state.df is not None:
    with st.expander("Step 2: Column Removal", expanded=True):
        if st.session_state.df_after_removal is None:
            with st.spinner("Analyzing columns for removal..."):
                try:
                    df_after_removal, remove_summary, removed_columns = remove_unwanted_columns(st.session_state.df, llm)
                    # Ensure removed_columns is a list
                    removed_columns = removed_columns if removed_columns is not None else []
                    st.session_state.df_after_removal = df_after_removal
                    st.session_state.remove_summary = remove_summary
                    st.session_state.removed_columns = removed_columns
                    st.session_state.columns_removed = False  # waiting for user confirmation
                except Exception as e:
                    st.error(f"Error during column analysis: {e}")
                    st.stop()

        if st.session_state.removed_columns:
            st.markdown(f"**Suggested columns to remove:** {', '.join(st.session_state.removed_columns)}")
        else:
            st.markdown("**No columns suggested for removal**")

        st.markdown(f"**Reason:** {st.session_state.remove_summary}")

        all_columns = list(st.session_state.df.columns)
        manual_columns_to_remove = st.multiselect(
            "Or manually select columns to remove",
            options=all_columns,
            help="Select additional columns you want to remove"
        )
        final_columns_to_remove = list(set(st.session_state.removed_columns + manual_columns_to_remove))
        if final_columns_to_remove:
            st.markdown(f"**Final columns to be removed:** {', '.join(final_columns_to_remove)}")
        st.dataframe(st.session_state.df.head(), height=200)

        if st.button("Remove Columns"):
            try:
                if not final_columns_to_remove:
                    st.session_state.df_after_removal = st.session_state.df
                    st.session_state.removed_columns = []
                    st.info("No columns were removed - proceeding with original dataset")
                else:
                    st.session_state.df_after_removal = st.session_state.df.drop(columns=final_columns_to_remove)
                    st.session_state.removed_columns = final_columns_to_remove
                    st.success(f"Successfully removed columns: {', '.join(final_columns_to_remove)}")
                st.session_state.columns_removed = True
                st.dataframe(st.session_state.df_after_removal.head(), height=200)
                st.session_state.expand_data_cleaning = True
            except Exception as e:
                st.error(f"Error processing columns: {e}")

# ----------------------------- Step 3: Data Cleaning -----------------------------
if st.session_state.df_after_removal is not None and st.session_state.columns_removed:
    with st.expander("Step 3: Data Cleaning", expanded=st.session_state.expand_data_cleaning):
        try:
            with st.spinner("Cleaning data..."):
                df_cleaned, cleaning_summary = data_cleaning(st.session_state.df_after_removal, llm)
            st.dataframe(df_cleaned.head(), height=200)
            st.session_state.df_cleaned = df_cleaned
            st.session_state.cleaning_summary = cleaning_summary
        except Exception as e:
            st.error(f"Error during data cleaning: {e}")

# ----------------------------- Step 4: Exploratory Data Analysis (EDA) -----------------------------
if st.session_state.df_cleaned is not None:
    with st.expander("Step 4: Exploratory Data Analysis (EDA)", expanded=True):
        try:
            if st.session_state.eda_summary is None:
                with st.spinner("Performing EDA..."):
                    eda_results = perform_eda(st.session_state.df_cleaned, llm)
                st.session_state.eda_summary = eda_results.get('summary', 'No summary available')
                st.session_state.eda_results = eda_results
            else:
                st.info("EDA already performed - showing previous results")
                eda_results = st.session_state.eda_results

            # Main visualization
            if eda_results.get('main_visualization'):
                main_vis_path = eda_results['main_visualization']
                if os.path.exists(main_vis_path):
                    try:
                        image = Image.open(main_vis_path)
                        st.image(image, use_container_width=True, caption="Main EDA Visualization")
                    except Exception as e:
                        st.error(f"Error opening main EDA image: {e}")
                else:
                    st.error(f"Main visualization file not found: {main_vis_path}")

            # All visualizations
            st.subheader("All Visualizations")
            visualizations = eda_results.get('visualizations', [])
            if visualizations:
                num_cols = 2
                cols = st.columns(num_cols)
                for i, vis_path in enumerate(visualizations):
                    try:
                        if os.path.exists(vis_path):
                            with cols[i % num_cols]:
                                image = Image.open(vis_path)
                                st.image(image, use_container_width=True, caption=f"Visualization {i+1}")
                        else:
                            st.error(f"Visualization file not found: {vis_path}")
                    except Exception as e:
                        st.error(f"Error displaying visualization {i+1}: {e}")

            st.subheader("Detailed EDA Summary")
            st.markdown("### Data Overview")
            st.write(st.session_state.df_cleaned.describe())
            st.markdown("### Missing Values")
            st.write(st.session_state.df_cleaned.isnull().sum())
            st.markdown("### LLM Analysis")
            st.write(eda_results.get('summary', 'No summary available'))

            if st.button("Proceed to Target Selection"):
                st.session_state.expand_target_selection = True

        except Exception as e:
            st.error(f"Error during EDA: {e}")

# ----------------------------- Step 5: Target Column Selection -----------------------------
if st.session_state.df_cleaned is not None and st.session_state.expand_target_selection:
    with st.expander("Step 5: Target Column Selection", expanded=True):
        try:
            df_cleaned = st.session_state.df_cleaned
            suggested_target = select_target_column(df_cleaned, llm)
            st.info(f"Suggested target column: **{suggested_target}**")
        except Exception as e:
            st.error(f"Error suggesting target column: {e}")
            df_cleaned = st.session_state.df_cleaned
            suggested_target = df_cleaned.columns[0]
        columns_list = list(df_cleaned.columns)
        try:
            default_index = columns_list.index(suggested_target)
        except ValueError:
            default_index = 0
        target_column = st.selectbox("Select Target Column", options=columns_list, index=default_index)
        if st.button("Accept Target Column"):
            if target_column in df_cleaned.columns and not df_cleaned[target_column].isnull().all() and df_cleaned[target_column].nunique() > 1:
                st.session_state.target_selected = target_column
                st.success(f"Target column '{target_column}' accepted.")
            else:
                st.error("Invalid target column selection.")

# ----------------------------- Step 6: Feature Engineering -----------------------------
if st.session_state.target_selected is not None:
    with st.expander("Step 6: Feature Engineering", expanded=True):
        try:
            with st.spinner("Engineering features..."):
                df_engineered, feature_summary = feature_engineering(
                    st.session_state.df_cleaned, llm, st.session_state.target_selected
                )
            st.dataframe(df_engineered.head(), height=200)
            st.session_state.df_engineered = df_engineered
            st.session_state.feature_summary = feature_summary
        except Exception as e:
            st.error(f"Error during Feature Engineering: {e}")

# ----------------------------- Step 7: Model Training & Evaluation -----------------------------
if st.session_state.df_engineered is not None:
    with st.expander("Step 7: Model Training & Evaluation", expanded=True):
        if st.session_state.training_completed is None or not st.session_state.training_completed:
            try:
                with st.spinner("Training model..."):
                    best_model, _, X, y, summary, best_metrics = model_selection_train_and_evaluation(
                        st.session_state.df_engineered, st.session_state.target_selected, llm
                    )
                st.session_state.best_model = best_model
                st.session_state.summary = summary
                st.session_state.best_metrics = best_metrics
                st.session_state.training_completed = True
            except Exception as e:
                st.error(f"Error during Model Training: {e}")
        if st.session_state.training_completed:
            st.markdown("#### Model Training Summary")
            st.json(st.session_state.best_metrics)
            if st.button("Retrain Model"):
                st.session_state.training_completed = False
                st.experimental_rerun()

# ----------------------------- Step 8: Store Summaries in DB -----------------------------
if all(st.session_state.get(key) is not None for key in 
       ["summary", "remove_summary", "cleaning_summary", "eda_summary", "feature_summary", "best_metrics", "file_name"]):
    with st.expander("Step 8: Store Summaries in DB", expanded=True):
        try:
            with st.spinner("Storing summaries in the database..."):
                store_summary_in_db(
                    summary=st.session_state.summary,
                    remove_summary=st.session_state.remove_summary,
                    cleaning_summary=st.session_state.cleaning_summary,
                    eda_summary=st.session_state.eda_summary,
                    feature_summary=st.session_state.feature_summary,
                    best_metrics=st.session_state.best_metrics,
                    file_name=st.session_state.file_name
                )
            st.success("Summaries stored successfully!")
        except Exception as e:
            st.error(f"Error storing summaries: {e}")

# ----------------------------- Step 9: Downloads -----------------------------
if st.session_state.df_cleaned is not None and st.session_state.best_model is not None:
    with st.expander("Step 9: Downloads", expanded=True):
        try:
            cleaned_data_path = os.path.join(os.getcwd(), "files_and_models", "processed_data", "cleaned_data.csv")
            model_path = os.path.join(os.getcwd(), "files_and_models", "saved_models", "best_model.pkl")
            os.makedirs(os.path.dirname(cleaned_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            st.session_state.df_cleaned.to_csv(cleaned_data_path, index=False)
            joblib.dump(st.session_state.best_model, model_path)
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download Cleaned Data",
                    data=open(cleaned_data_path, "rb").read(),
                    file_name="cleaned_data.csv",
                    mime="text/csv"
                )
            with col2:
                st.download_button(
                    "Download Best Model",
                    data=open(model_path, "rb").read(),
                    file_name="best_model.pkl",
                    mime="application/octet-stream"
                )
        except Exception as e:
            st.error(f"Error during downloads: {e}")

# ----------------------------- Divider before Chat Section -----------------------------
st.markdown("---")

# ----------------------------- Step 10: Chatbot -----------------------------
with st.expander("Step 10: Chatbot", expanded=True):
    st.markdown("### Chat with ML Assistant")
    st.markdown("Ask questions about your ML workflow (e.g., data preprocessing, feature engineering, model evaluation).")

    # Create a container for chat history
    chat_history_container = st.container()

    # Display chat history using st.chat_message for a consistent look
    if st.session_state.chat_histories.get(st.session_state.current_file):
        for msg in st.session_state.chat_histories[st.session_state.current_file]:
            if msg["role"] == "user":
                chat_history_container.chat_message("user").write(f"{msg['content']} ({msg['timestamp']})")
            else:
                chat_history_container.chat_message("assistant").write(f"{msg['content']} ({msg['timestamp']})")
    else:
        chat_history_container.info("No chat history yet. Ask a question below!")

    # Place the input box at the bottom
    user_prompt = st.chat_input("Your message here...")
    if user_prompt:
        timestamp = time.strftime("%H:%M:%S")
        # Append and display the user's message in the chat history container
        st.session_state.chat_histories.setdefault(st.session_state.current_file, []).append({
            "role": "user",
            "content": user_prompt,
            "timestamp": timestamp
        })
        chat_history_container.chat_message("user").write(f"{user_prompt} ({timestamp})")

        # Process the assistant's response
        with chat_history_container.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                with st.spinner("Analyzing query and retrieving context..."):
                    # Retrieve context using RAG
                    try:
                        rag_response = answer_queries_using_rag([user_prompt], llm, settings.postgresql_url)
                        retrieved_context = rag_response.get(user_prompt, "No relevant context found")
                        st.session_state.last_retrieved_context = retrieved_context
                        messages = [
                            SystemMessage(
                                content=(
                                    "You are an AI assistant specialized in machine learning workflows. "
                                    "Provide direct, concise answers using bullet points, focusing on the most relevant details. "
                                    "Keep responses under 50 words unless additional context is required for clarity. "
                                    "Avoid including code examples or generic information."

                                )
                            ),
                            HumanMessage(
                                content=f"User Query: {user_prompt}\n\nRetrieved Context:\n{retrieved_context}\n\n"
                                "Please provide a concise answer based on the context."

                            )
                        ]
                        st.session_state.rag_status = "success"
                    except Exception as e:
                        st.error("Error retrieving context. Proceeding with general guidance.")
                        st.session_state.rag_status = "error"
                        messages = [
                            SystemMessage(
                                content="You are an AI assistant specialized in machine learning workflows."
                            ),
                            HumanMessage(
                                content=f"User Query: {user_prompt}\n\nContext retrieval failed; provide general guidance."
                            )
                        ]

                    # Generate the assistant's response using the LLM
                    assistant_response = llm.invoke(messages)
                    full_response = assistant_response.content.strip()

                # Simulate a typing effect by revealing the response character-by-character
                displayed_text = ""
                for char in full_response:
                    displayed_text += char
                    message_placeholder.markdown(f"**Assistant:** {displayed_text}▌")
                    time.sleep(0.02)
                message_placeholder.markdown(f"**Assistant ({timestamp}):** {full_response}")

                # Append the assistant's response to chat history
                st.session_state.chat_histories[st.session_state.current_file].append({
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": timestamp
                })
            except Exception as e:
                error_msg = f"Error generating response: {e}"
                st.error(error_msg)
                st.session_state.chat_histories[st.session_state.current_file].append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": timestamp
                })
