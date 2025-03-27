from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlmodel import Field, SQLModel, create_engine, Session, select, JSON
from sqlalchemy import or_, text, func
import logging
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the SQLModel engine using the PostgreSQL URI from config
engine = create_engine(settings.postgresql_url, echo=True)

class Summary(SQLModel, table=True):
    """Table for storing machine learning summaries and metrics"""
    __tablename__ = "ml_summaries"
    __table_args__ = {'extend_existing': True}

    id: Optional[int] = Field(default=None, primary_key=True)
    file_name: str = Field(nullable=False)
    summary: Dict[str, Any] = Field(default={}, sa_type=JSON)
    remove_summary: Optional[Dict[str, Any]] = Field(default=None, sa_type=JSON)
    cleaning_summary: Optional[Dict[str, Any]] = Field(default=None, sa_type=JSON)
    eda_summary: Optional[Dict[str, Any]] = Field(default=None, sa_type=JSON)
    feature_summary: Optional[Dict[str, Any]] = Field(default=None, sa_type=JSON)
    best_metrics: Optional[Dict[str, Any]] = Field(default=None, sa_type=JSON)
    all_models_metrics: Optional[Dict[str, Any]] = Field(default={}, sa_type=JSON)


class QA(SQLModel, table=True):
    """Table for storing question-answer pairs"""
    __table_args__ = {'extend_existing': True}
    id: Optional[int] = Field(default=None, primary_key=True)
    question: str = Field(nullable=False)
    answer: str = Field(nullable=False)

def init_db() -> bool:
    """Initialize the database and create tables if they don't exist"""
    try:
        with Session(engine) as session:
            # Check if tables exist and create if needed
            SQLModel.metadata.create_all(engine)
            
            # Check and add missing columns
            from sqlalchemy import inspect
            inspector = inspect(engine)
            columns = [col['name'] for col in inspector.get_columns('ml_summaries')]
            
            if 'all_models_metrics' not in columns:
                session.execute("ALTER TABLE ml_summaries ADD COLUMN all_models_metrics JSONB")
                session.commit()
                
            return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False

def store_summary_in_db(
    summary: Dict[str, Any],
    remove_summary: Optional[Dict[str, Any]] = None,
    cleaning_summary: Optional[Dict[str, Any]] = None,
    eda_summary: Optional[Dict[str, Any]] = None,
    feature_summary: Optional[Dict[str, Any]] = None,
    best_metrics: Optional[Dict[str, Any]] = None,
    file_name: Optional[str] = None
) -> bool:
    """Store a summary and related data in the database"""
    if not init_db():
        logger.error("Failed to initialize database")
        return False
        
    try:
        with Session(engine) as session:
            summary_record = Summary(
                file_name=file_name or "unknown",
                summary=summary or {},
                remove_summary=remove_summary or {},
                cleaning_summary=cleaning_summary or {},
                eda_summary=eda_summary or {},
                feature_summary=feature_summary or {},
                best_metrics=best_metrics or {
                    "model_name": "",
                    "metrics": {},
                    "parameters": {}
                },
                all_models_metrics=summary.get('model_metrics', {}) or {}

            )
            session.add(summary_record)
            session.commit()
            return True
    except Exception as e:
        logger.error(f"Error storing summary: {e}")
        return False

def get_relevant_text(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieve the most relevant summaries for RAG context.
    Searches across all relevant fields and returns partial matches.
    """

    """
    Retrieve the most relevant summaries for RAG context.
    Searches across all relevant fields and returns partial matches.
    """
    try:
        with Session(engine) as session:
            from sqlalchemy import cast, String, or_
            

            search_fields = [
                "model_type", "best_model", "model_name", 
                "model_metrics", "all_models_metrics",
            ]

            
            # Build dynamic search conditions
            conditions = []
            for field in search_fields:
                if field in ["all_models_metrics", "model_metrics"]:
                    # Handle nested JSON fields
                    conditions.append(
                        cast(Summary.summary[field], String).ilike(f"%{query}%")
                    )
                else:
                    conditions.append(
                        cast(Summary.summary[field], String).ilike(f"%{query}%")
                    )
            
            # Also search in the dedicated metrics and feature engineering fields
            conditions.append(cast(Summary.best_metrics, String).ilike(f"%{query}%"))
            conditions.append(cast(Summary.all_models_metrics, String).ilike(f"%{query}%"))  # Search in all models metrics

            conditions.append(cast(Summary.feature_summary, String).ilike(f"%{query}%"))

            
            # Combine conditions with OR and get most recent matches
            statement = select(Summary).where(
                or_(*conditions)
            ).order_by(Summary.id.desc()).limit(limit)

            results = session.exec(statement).all()
            
            # Return all matching summaries with relevant data
            return [{  # Return all matching summaries with relevant data

                'remove_summary': result.remove_summary,
                'cleaning_summary': result.cleaning_summary,
                'feature_summary': result.feature_summary,
                'best_metrics': result.best_metrics,
                'all_models_metrics': result.all_models_metrics or {},
                'model_metrics': result.all_models_metrics or {}  # Backward compatibility
            } for result in results]





    except Exception as e:
        print(f"Error retrieving summaries: {e}")
