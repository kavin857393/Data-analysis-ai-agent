import psycopg2
from psycopg2.extras import RealDictCursor
from config import settings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def answer_queries_using_rag(queries, llm, postgresql_url):
    """
    Answer queries using RAG, including model training details in the context.
    
    Args:
        queries: A list of user queries.
        llm: The language model to use.
        postgresql_url: Database connection URL.
        
    Returns:
        A dictionary mapping each query to its answer.
    """
    answers = {}
    
    # Build model training details from the LLM object
    if hasattr(llm, 'model_training_summary'):
        model_details = "\nModel Training Details:\n"
        # Best model information
        if 'best_model' in llm.model_training_summary:
            best_model = llm.model_training_summary['best_model']
            model_details += f"Best Model: {best_model['model_name']}\n"
            model_details += f"Accuracy: {best_model.get('accuracy', 'N/A')}\n"
            model_details += f"Parameters: {best_model.get('parameters', {})}\n\n"
        
        # All models metrics
        if 'model_metrics' in llm.model_training_summary:
            model_details += "All Trained Models:\n"
            for model_name, metrics in llm.model_training_summary['model_metrics'].items():
                model_details += (
                    f"- {model_name}: Accuracy={metrics.get('accuracy', 'N/A')}, "
                    f"Precision={metrics.get('precision', 'N/A')}, "
                    f"Recall={metrics.get('recall', 'N/A')}, "
                    f"F1 Score={metrics.get('f1_score', 'N/A')}\n"
                )
        answers['model_details'] = model_details

    # Retrieve the most recent summary record from PostgreSQL with additional metadata
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(postgresql_url)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT *, 
                   (SELECT COUNT(*) FROM ml_summaries) as total_runs,
                   all_models_metrics
            FROM ml_summaries 
            ORDER BY id DESC 
            LIMIT 1
        """)
        summary_record = cur.fetchone()
        
        # Build all models information from the summary record
        if summary_record and 'all_models_metrics' in summary_record:
            # print("Summary Record Structure:", summary_record.keys())
            # print("All Models Metrics Structure:", type(summary_record['all_models_metrics']))

            
            models_info = "\nAll Trained Models and Metrics:\n"
            for model_data in summary_record['all_models_metrics']:
                # print("Model Data Structure:", model_data.keys())

                models_info += f"\nModel Name: {model_data['name']}\n"
                # Check for metrics in either 'metrics' or 'best_metrics'
                metrics = model_data.get('metrics', model_data.get('best_metrics', {}))
                if metrics:
                    models_info += "• Metrics:\n"
                    for metric, value in metrics.items():
                        models_info += f"  - {metric}: {value:.4f}\n"
                else:
                    models_info += "• Metrics: N/A\n"
                # Parameters if available
                if 'parameters' in model_data and model_data['parameters']:
                    models_info += "• Parameters:\n"
                    for param, value in model_data['parameters'].items():
                        models_info += f"  - {param}: {value}\n"
                models_info += "----------------------------\n"
            answers['all_models'] = models_info

    except Exception as e:
        for query in queries:
            answers[query] = f"Error retrieving summary from database: {e}"
        return answers
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

    # Build a combined context string including both best and all models metrics
    if not summary_record:
        combined_context = "No summaries available in the database."
    else:
        cleaning_summary = summary_record.get('cleaning_summary', {})
        feature_summary = summary_record.get('feature_summary', {})
        combined_context = (
            f"Data Cleaning Steps:\n"
            f"- Missing Data Handling: {cleaning_summary.get('missing_data', 'Not specified')}\n"
            f"- Outlier Treatment: {cleaning_summary.get('outliers', 'Not specified')}\n"
            f"- Data Normalization: {cleaning_summary.get('normalization', 'Not specified')}\n\n"
            f"Feature Engineering Steps:\n"
            f"- New Features: {feature_summary.get('new_features', 'None created')}\n"
            f"- Feature Transformations: {feature_summary.get('transformations', 'None applied')}\n"
            f"- Feature Selection: {feature_summary.get('selection', 'All features used')}\n\n"
            f"Best Model Metrics:\n{summary_record.get('best_metrics', 'Not available')}\n\n"
            f"All Models Metrics:\n{summary_record.get('all_models_metrics', 'Not available')}\n"
        )

    # Split the combined context into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        keep_separator=True
    )
    chunks = text_splitter.split_text(combined_context)
    chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]

    # Initialize the embeddings model
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    chunk_embeddings = embeddings_model.embed_documents(chunks)
    chunk_embeddings_array = np.array(chunk_embeddings)
    
    for query in queries:
        query_embedding = embeddings_model.embed_query(query)
        query_embedding = np.array(query_embedding).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, chunk_embeddings_array)[0]
        top_indices = np.argsort(similarities)[-3:]
        top_chunks = [chunks[i] for i in top_indices]
        relevant_context = "\n".join(top_chunks)
        
        # Build the prompt with all available context
        prompt = f"""You are an AI assistant for machine learning workflow analysis. 
Provide direct, concise answers to the query using only the available information, ensuring clarity and precision.

Query: {query}

Available Context:
{relevant_context}

Model Training Details:
{answers.get('model_details', 'No model details available')}

All Trained Models:
{answers.get('all_models', 'No model information available')}

Instructions:
1. Answer the query directly using only the provided context, focusing on the most relevant details.
2. Use bullet points for structured responses.
3. Do not include code examples or implementation details.
4. Do not provide generic information, analysis, or suggestions.
5. If no specific feature engineering steps are mentioned, state "No feature engineering details available."
6. For feature engineering queries:
   - List only the specific steps performed in this workflow.
   - Include only what is explicitly documented in the context.
   - Do not add any general information about feature engineering.
   - If steps are listed, present them exactly as documented.
7. Focus on providing precise, factual information from the workflow.
"""
        try:
            response = llm.invoke(prompt)
            answers[query] = response.content.strip()
        except Exception as e:
            answers[query] = f"Error generating response: {e}"
    
    return answers
