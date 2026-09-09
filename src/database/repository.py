"""Repository layer for database operations."""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy.orm import Session

from src.database.schema import (
    ModelRegistry,
    Outcome,
    Prediction,
    Recommendation,
    RiskScore,
    RLAction,
    RLEpisode,
    SecurityEvent,
    init_db,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class CyberRiskRepository:
    """Repository for all database operations."""

    def __init__(self, session: Optional[Session] = None):
        if session is None:
            Session = init_db()
            self.session = Session()
        else:
            self.session = session

    # --- Security Events ---

    def add_event(self, event_data: Dict[str, Any]) -> SecurityEvent:
        """Add a security event to the database.

        Args:
            event_data: Dictionary of event data.

        Returns:
            The created SecurityEvent.
        """
        event = SecurityEvent(**event_data)
        self.session.add(event)
        self.session.commit()
        return event

    def add_events_bulk(self, events_df: pd.DataFrame) -> int:
        """Add multiple security events in bulk.

        Args:
            events_df: DataFrame of events.

        Returns:
            Number of events added.
        """
        count = 0
        for _, row in events_df.iterrows():
            event_data = row.to_dict()
            # Convert non-serializable values
            for key, value in event_data.items():
                if hasattr(value, "isoformat"):
                    event_data[key] = value.isoformat()
            self.add_event(event_data)
            count += 1
        return count

    def get_event(self, event_id: int) -> Optional[SecurityEvent]:
        """Get a security event by ID.

        Args:
            event_id: Event ID.

        Returns:
            SecurityEvent or None.
        """
        return self.session.query(SecurityEvent).filter(SecurityEvent.id == event_id).first()

    def get_events(
        self,
        limit: int = 100,
        offset: int = 0,
        source_type: Optional[str] = None,
        risk_label: Optional[str] = None,
    ) -> List[SecurityEvent]:
        """Get security events with optional filters.

        Args:
            limit: Maximum number of events.
            offset: Pagination offset.
            source_type: Filter by source type.
            risk_label: Filter by risk label.

        Returns:
            List of SecurityEvent objects.
        """
        query = self.session.query(SecurityEvent)
        if source_type:
            query = query.filter(SecurityEvent.source_type == source_type)
        if risk_label:
            query = query.filter(SecurityEvent.risk_label == risk_label)
        return query.order_by(SecurityEvent.timestamp.desc()).limit(limit).offset(offset).all()

    # --- Predictions ---

    def add_prediction(
        self,
        event_id: int,
        model_name: str,
        prediction: str,
        probability: float,
    ) -> Prediction:
        """Add a model prediction.

        Args:
            event_id: Event ID.
            model_name: Name of the model.
            prediction: Predicted label.
            probability: Prediction probability.

        Returns:
            The created Prediction.
        """
        pred = Prediction(
            event_id=event_id,
            model_name=model_name,
            prediction=prediction,
            probability=probability,
        )
        self.session.add(pred)
        self.session.commit()
        return pred

    # --- Risk Scores ---

    def add_risk_score(
        self,
        event_id: int,
        risk_score: float,
        risk_level: str,
        supervised_score: float,
        anomaly_score: float,
        contextual_score: float,
        explanation: Optional[str] = None,
    ) -> RiskScore:
        """Add a risk score for an event.

        Args:
            event_id: Event ID.
            risk_score: Fused risk score.
            risk_level: Risk level category.
            supervised_score: Supervised model score.
            anomaly_score: Anomaly detection score.
            contextual_score: Contextual indicator score.
            explanation: Risk explanation text.

        Returns:
            The created RiskScore.
        """
        score = RiskScore(
            event_id=event_id,
            risk_score=risk_score,
            risk_level=risk_level,
            supervised_score=supervised_score,
            anomaly_score=anomaly_score,
            contextual_score=contextual_score,
            explanation=explanation,
        )
        self.session.add(score)
        self.session.commit()
        return score

    # --- Recommendations ---

    def add_recommendation(
        self,
        event_id: int,
        action: str,
        score: float,
        description: Optional[str] = None,
        method: str = "hybrid",
    ) -> Recommendation:
        """Add a recommendation for an event.

        Args:
            event_id: Event ID.
            action: Recommended action.
            score: Recommendation score.
            description: Action description.
            method: Recommendation method.

        Returns:
            The created Recommendation.
        """
        rec = Recommendation(
            event_id=event_id,
            action=action,
            score=score,
            description=description,
            method=method,
        )
        self.session.add(rec)
        self.session.commit()
        return rec

    # --- RL Episodes ---

    def add_rl_episode(
        self,
        episode_number: int,
        total_reward: float,
        episode_length: int,
        success: bool,
        epsilon: float,
    ) -> RLEpisode:
        """Add an RL training episode.

        Args:
            episode_number: Episode number.
            total_reward: Total reward for the episode.
            episode_length: Number of steps.
            success: Whether the episode was successful.
            epsilon: Exploration rate at the time.

        Returns:
            The created RLEpisode.
        """
        episode = RLEpisode(
            episode_number=episode_number,
            total_reward=total_reward,
            episode_length=episode_length,
            success=success,
            epsilon=epsilon,
        )
        self.session.add(episode)
        self.session.commit()
        return episode

    def add_rl_action(
        self,
        episode_id: int,
        step: int,
        state: Dict[str, Any],
        action: str,
        reward: float,
        terminated: bool,
    ) -> RLAction:
        """Add an RL action record.

        Args:
            episode_id: Episode ID.
            step: Step number.
            state: State vector.
            action: Action name.
            reward: Reward received.
            terminated: Whether the episode terminated.

        Returns:
            The created RLAction.
        """
        rl_action = RLAction(
            episode_id=episode_id,
            step=step,
            state=state,
            action=action,
            reward=reward,
            terminated=terminated,
        )
        self.session.add(rl_action)
        self.session.commit()
        return rl_action

    # --- Outcomes ---

    def add_outcome(
        self,
        recommendation_id: int,
        event_id: int,
        action_taken: str,
        outcome: str,
        notes: Optional[str] = None,
    ) -> Outcome:
        """Add an outcome for a recommendation.

        Args:
            recommendation_id: Recommendation ID.
            event_id: Event ID.
            action_taken: Action that was taken.
            outcome: Outcome (success, failure, partial).
            notes: Additional notes.

        Returns:
            The created Outcome.
        """
        outcome_record = Outcome(
            recommendation_id=recommendation_id,
            event_id=event_id,
            action_taken=action_taken,
            outcome=outcome,
            notes=notes,
        )
        self.session.add(outcome_record)
        self.session.commit()
        return outcome_record

    # --- Model Registry ---

    def register_model(
        self,
        model_name: str,
        model_type: str,
        model_path: str,
        metrics: Optional[Dict[str, Any]] = None,
        is_active: bool = False,
    ) -> ModelRegistry:
        """Register a trained model.

        Args:
            model_name: Name of the model.
            model_type: Type of model.
            model_path: Path to the saved model.
            metrics: Model performance metrics.
            is_active: Whether this is the active model.

        Returns:
            The created ModelRegistry entry.
        """
        # Deactivate other models of the same type if this is active
        if is_active:
            self.session.query(ModelRegistry).filter(
                ModelRegistry.model_type == model_type
            ).update({"is_active": False})

        model = ModelRegistry(
            model_name=model_name,
            model_type=model_type,
            model_path=model_path,
            metrics=metrics or {},
            is_active=is_active,
        )
        self.session.add(model)
        self.session.commit()
        return model

    def get_active_model(self, model_type: str) -> Optional[ModelRegistry]:
        """Get the active model of a given type.

        Args:
            model_type: Type of model.

        Returns:
            ModelRegistry entry or None.
        """
        return (
            self.session.query(ModelRegistry)
            .filter(ModelRegistry.model_type == model_type, ModelRegistry.is_active == True)
            .first()
        )

    # --- Query Helpers ---

    def get_risk_score_distribution(self) -> Dict[str, int]:
        """Get the distribution of risk levels.

        Returns:
            Dictionary of risk level to count.
        """
        results = (
            self.session.query(RiskScore.risk_level, RiskScore.id)
            .all()
        )
        distribution: Dict[str, int] = {}
        for level, _ in results:
            distribution[level] = distribution.get(level, 0) + 1
        return distribution

    def get_recent_events_with_risk(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent events with their risk scores.

        Args:
            limit: Maximum number of events.

        Returns:
            List of dictionaries with event and risk data.
        """
        events = self.get_events(limit=limit)
        result = []
        for event in events:
            risk = (
                self.session.query(RiskScore)
                .filter(RiskScore.event_id == event.id)
                .order_by(RiskScore.computed_at.desc())
                .first()
            )
            result.append({
                "event_id": event.id,
                "timestamp": event.timestamp,
                "source_type": event.source_type,
                "event_type": event.event_type,
                "risk_label": event.risk_label,
                "risk_score": risk.risk_score if risk else None,
                "risk_level": risk.risk_level if risk else None,
            })
        return result

    def close(self) -> None:
        """Close the database session."""
        self.session.close()