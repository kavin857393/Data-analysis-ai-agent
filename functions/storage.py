# functions/storage.py

import psycopg2
from psycopg2.extras import Json
from config import settings
import streamlit as st

def store_summary_in_db(summary, remove_summary=None, cleaning_summary=None, eda_summary=None, feature_summary=None, best_metrics=None, file_name=None):
    try:
        conn = psycopg2.connect(settings.postgresql_url)
        cur = conn.cursor()
        
        # Extract all models' metrics and parameters from the summary
        all_models_metrics = summary.get('model_metrics', {})
        all_models_params = summary.get('best_hyperparameters', {})
        
        # Prepare models data for storage
        models_data = []
        for model_name, metrics in all_models_metrics.items():
            model_info = {
                'name': model_name,
                'accuracy': metrics.get('accuracy'),
                'precision': metrics.get('precision'),
                'recall': metrics.get('recall'),
                'f1_score': metrics.get('f1_score')
            }
            models_data.append(model_info)

        insert_query = """
            INSERT INTO ml_summaries
                (file_name, summary, remove_summary, cleaning_summary, eda_summary, feature_summary, best_metrics, all_models_metrics)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        cur.execute(
            insert_query,
            (
                file_name or "unknown",
                Json(summary),
                Json(remove_summary) if remove_summary is not None else None,
                Json(cleaning_summary) if cleaning_summary is not None else None,
                Json(eda_summary) if eda_summary is not None else None,
                Json(feature_summary) if feature_summary is not None else None,
                Json(best_metrics) if best_metrics is not None else None,
                Json(models_data)  # Store all models' metrics and parameters
            ),
        )

        conn.commit()
        st.success("Summaries stored in PostgreSQL successfully.")
    except Exception as e:
        st.error(f"Error storing summaries: {e}")
    finally:
        cur.close()
        conn.close()
