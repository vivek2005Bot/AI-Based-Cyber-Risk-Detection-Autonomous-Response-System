"""SQLAlchemy database schema for the cyber risk detection system."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

Base = declarative_base()


class SecurityEvent(Base):
    """A security event record from any data source."""

    __tablename__ = "security_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    source_type = Column(String(50), index=True)
    event_type = Column(String(50), index=True)
    user_id = Column(String(100), nullable=True, index=True)
    host_id = Column(String(100), nullable=True, index=True)
    source_ip = Column(String(50), nullable=True)
    destination_ip = Column(String(50), nullable=True)
    risk_label = Column(String(20), index=True)
    attack_type = Column(String(50), nullable=True)
    features = Column(JSON, default=dict)
    raw_data = Column(JSON, default=dict)

    predictions = relationship("Prediction", back_populates="event", cascade="all, delete-orphan")


class Prediction(Base):
    """A model prediction for a security event."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey("security_events.id"), index=True)
    model_name = Column(String(100), index=True)
    prediction = Column(String(20))
    probability = Column(Float)
    predicted_at = Column(DateTime, default=datetime.utcnow)

    event = relationship("SecurityEvent", back_populates="predictions")


class RiskScore(Base):
    """A fused risk score for a security event."""

    __tablename__ = "risk_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey("security_events.id"), index=True)
    risk_score = Column(Float)
    risk_level = Column(String(20), index=True)
    supervised_score = Column(Float)
    anomaly_score = Column(Float)
    contextual_score = Column(Float)
    explanation = Column(Text, nullable=True)
    computed_at = Column(DateTime, default=datetime.utcnow)


class Recommendation(Base):
    """A recommended response action for a security event."""

    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey("security_events.id"), index=True)
    action = Column(String(50))
    description = Column(Text, nullable=True)
    score = Column(Float)
    method = Column(String(50), default="hybrid")
    recommended_at = Column(DateTime, default=datetime.utcnow)


class RLEpisode(Base):
    """A reinforcement learning training episode."""

    __tablename__ = "rl_episodes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    episode_number = Column(Integer, index=True)
    total_reward = Column(Float)
    episode_length = Column(Integer)
    success = Column(Boolean, default=False)
    epsilon = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class RLAction(Base):
    """An action taken by the RL agent during training."""

    __tablename__ = "rl_actions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    episode_id = Column(Integer, ForeignKey("rl_episodes.id"), index=True)
    step = Column(Integer)
    state = Column(JSON, default=dict)
    action = Column(String(50))
    reward = Column(Float)
    terminated = Column(Boolean, default=False)


class Outcome(Base):
    """The outcome of a recommended action."""

    __tablename__ = "outcomes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    recommendation_id = Column(Integer, ForeignKey("recommendations.id"), index=True)
    event_id = Column(Integer, ForeignKey("security_events.id"), index=True)
    action_taken = Column(String(50))
    outcome = Column(String(20))  # success, failure, partial
    notes = Column(Text, nullable=True)
    recorded_at = Column(DateTime, default=datetime.utcnow)


class ModelRegistry(Base):
    """Registry of trained models."""

    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), index=True)
    model_type = Column(String(50))  # supervised, ensemble, anomaly
    model_path = Column(String(500))
    metrics = Column(JSON, default=dict)
    trained_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=False)


def init_db(db_url: Optional[str] = None) -> Any:
    """Initialize the database and return a session factory.

    Args:
        db_url: Database URL. If None, uses config value.

    Returns:
        SQLAlchemy session factory.
    """
    config = get_config()
    if db_url is None:
        db_type = config.get("database.type", "sqlite")
        if db_type == "sqlite":
            sqlite_path = config.get("database.sqlite_path", "data/cyber_risk.db")
            Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
            db_url = f"sqlite:///{sqlite_path}"
        else:
            pg = config.get_section("database.postgresql")
            db_url = (
                f"postgresql://{pg.get('username', 'postgres')}:{pg.get('password', '')}"
                f"@{pg.get('host', 'localhost')}:{pg.get('port', 5432)}/{pg.get('database', 'cyber_risk')}"
            )

    logger.info(f"Initializing database at {db_url}")
    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session