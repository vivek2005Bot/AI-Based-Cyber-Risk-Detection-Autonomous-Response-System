"""FastAPI application for the AI-Based Cyber Risk Detection System."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.anomaly.detectors import AnomalyDetector
from src.database.repository import CyberRiskRepository
from src.models.supervised import ModelTrainer, SupervisedModelFactory
from src.recommendation.hybrid import HybridRecommender
from src.risk_fusion.fusion import RiskFusionEngine
from src.rl.q_learning import QLearningAgent
from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)
config = get_config()

app = FastAPI(
    title="AI-Based Cyber Risk Detection System",
    description="API for cyber risk detection, anomaly identification, risk scoring, and response recommendation.",
    version="1.0.0",
)

# CORS
cors_origins = config.get("api.cors_origins", ["http://localhost:3000", "http://localhost:8501"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
repository = CyberRiskRepository()
model_factory = SupervisedModelFactory()
model_trainer = ModelTrainer()
anomaly_detector = AnomalyDetector()
risk_fusion = RiskFusionEngine()
recommender = HybridRecommender()
rl_agent = QLearningAgent()

# In-memory model store (in production, load from disk/database)
trained_models: Dict[str, Any] = {}


# --- Pydantic Models ---

class EventData(BaseModel):
    """Security event data for prediction."""
    features: Dict[str, Any]
    source_type: str = "synthetic"
    event_type: str = "auth"
    user_id: Optional[str] = None
    host_id: Optional[str] = None
    source_ip: Optional[str] = None


class RiskAssessmentRequest(BaseModel):
    """Request for a full risk assessment."""
    features: Dict[str, Any]
    source_type: str = "synthetic"
    event_type: str = "auth"
    user_id: Optional[str] = None
    host_id: Optional[str] = None
    source_ip: Optional[str] = None
    cvss_score: Optional[float] = None
    asset_criticality: Optional[float] = None
    threat_score: Optional[float] = None
    incident_count: Optional[int] = None
    hours_since_event: Optional[float] = None


class RLTrainingRequest(BaseModel):
    """Request to train the RL agent."""
    episodes: int = 1000
    evaluate: bool = True


# --- Health ---

@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models_loaded": len(trained_models),
    }


# --- Prediction ---

@app.post("/predict")
def predict(event: EventData) -> Dict[str, Any]:
    """Make a prediction for a security event.

    Args:
        event: Event data with features.

    Returns:
        Dictionary with predictions from all available models.
    """
    if not trained_models:
        raise HTTPException(status_code=503, detail="No trained models available. Run the training pipeline first.")

    # Convert features to DataFrame
    df = pd.DataFrame([event.features])

    predictions = {}
    for name, model in trained_models.items():
        try:
            pred = model.predict(df.values)[0]
            prob = None
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(df.values)[0].tolist()
            predictions[name] = {
                "prediction": str(pred),
                "probability": prob,
            }
        except Exception as e:
            logger.error(f"Prediction failed for {name}: {e}")
            predictions[name] = {"error": str(e)}

    return {
        "event_id": None,
        "predictions": predictions,
    }


# --- Risk Assessment ---

@app.post("/assess-risk")
def assess_risk(request: RiskAssessmentRequest) -> Dict[str, Any]:
    """Perform a full risk assessment for a security event.

    Args:
        request: Risk assessment request.

    Returns:
        Dictionary with risk score, level, and explanation.
    """
    # Get supervised prediction score
    supervised_score = 0.5
    if trained_models:
        df = pd.DataFrame([request.features])
        scores = []
        for name, model in trained_models.items():
            try:
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(df.values)[0]
                    # Use probability of positive class (suspicious)
                    scores.append(float(prob[1] if prob.shape[0] > 1 else prob[0]))
                else:
                    pred = model.predict(df.values)[0]
                    scores.append(1.0 if str(pred) in ["suspicious", "1", "attack"] else 0.0)
            except Exception as e:
                logger.warning(f"Could not get prediction from {name}: {e}")
        if scores:
            supervised_score = float(np.mean(scores))

    # Get anomaly score
    try:
        feature_df = pd.DataFrame([request.features])
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            anomaly_score = float(anomaly_detector.combined_anomaly_score(feature_df[numeric_cols].values)[0])
        else:
            anomaly_score = 0.0
    except Exception as e:
        logger.warning(f"Anomaly detection failed: {e}")
        anomaly_score = 0.0

    # Get contextual score
    contextual_df = pd.DataFrame([{
        "cvss_score": request.cvss_score or 0,
        "asset_criticality": request.asset_criticality or 0,
        "threat_score": request.threat_score or 0,
        "incident_count": request.incident_count or 0,
        "hours_since_event": request.hours_since_event or 0,
    }])
    contextual_df = risk_fusion.contextual_engine.compute_indicators(contextual_df)
    contextual_score = float(risk_fusion.contextual_engine.combined_contextual_score(contextual_df)[0])

    # Fuse risk
    risk_score = float(risk_fusion.fuse(
        np.array([supervised_score]),
        np.array([anomaly_score]),
        np.array([contextual_score]),
    )[0])
    risk_level = risk_fusion.risk_level(np.array([risk_score]))[0]

    # Generate explanation
    explanation = risk_fusion.explain_risk(
        risk_score, supervised_score, anomaly_score, contextual_score
    )

    # Get recommendations
    event_for_rec = {
        "event_type": request.event_type,
        "risk_level": risk_level,
        "source_type": request.source_type,
        **request.features,
    }
    recommendations = recommender.recommend(event_for_rec, user_id=request.user_id, top_k=5)

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "supervised_score": supervised_score,
        "anomaly_score": anomaly_score,
        "contextual_score": contextual_score,
        "explanation": explanation,
        "recommendations": recommendations,
    }


# --- Anomaly Detection ---

@app.post("/detect-anomalies")
def detect_anomalies(features: Dict[str, Any]) -> Dict[str, Any]:
    """Detect anomalies in a feature set.

    Args:
        features: Dictionary of feature values.

    Returns:
        Dictionary with anomaly scores.
    """
    df = pd.DataFrame([features])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        raise HTTPException(status_code=400, detail="No numeric features provided.")

    X = df[numeric_cols].values
    results = anomaly_detector.run_all(X)

    output = {}
    for name, result in results.items():
        output[name] = {
            "is_anomaly": bool(result["predictions"][0] == -1),
            "anomaly_score": float(result["anomaly_scores"][0]),
        }

    return {"anomalies": output}


# --- Recommendations ---

@app.post("/recommend")
def get_recommendations(event: EventData) -> Dict[str, Any]:
    """Get response recommendations for a security event.

    Args:
        event: Event data.

    Returns:
        Dictionary with recommended actions.
    """
    event_dict = {
        "event_type": event.event_type,
        "source_type": event.source_type,
        "risk_level": "medium",  # Default; would be computed from risk assessment
        **event.features,
    }
    recommendations = recommender.recommend(event_dict, user_id=event.user_id, top_k=5)
    return {"recommendations": recommendations}


# --- RL ---

@app.post("/rl/train")
def train_rl(request: RLTrainingRequest) -> Dict[str, Any]:
    """Train the Q-Learning agent.

    Args:
        request: Training request.

    Returns:
        Dictionary with training metrics.
    """
    global rl_agent
    rl_agent = QLearningAgent()
    metrics = rl_agent.train(n_episodes=request.episodes)

    if request.evaluate:
        eval_metrics = rl_agent.evaluate(n_episodes=100)
        metrics["evaluation"] = eval_metrics

    return metrics


@app.get("/rl/status")
def rl_status() -> Dict[str, Any]:
    """Get the status of the RL agent.

    Returns:
        Dictionary with RL agent status.
    """
    return {
        "trained": len(rl_agent.q_table) > 0,
        "q_table_size": len(rl_agent.q_table),
        "epsilon": rl_agent.epsilon,
        "episodes_completed": len(rl_agent.episode_rewards),
        "avg_reward": float(np.mean(rl_agent.episode_rewards[-100:])) if rl_agent.episode_rewards else 0.0,
    }


@app.get("/rl/policy")
def rl_policy() -> Dict[str, Any]:
    """Get the learned policy.

    Returns:
        Dictionary with the learned policy.
    """
    return {"policy": rl_agent.get_policy()}


# --- Models ---

@app.get("/models")
def list_models() -> Dict[str, Any]:
    """List available and trained models.

    Returns:
        Dictionary with model information.
    """
    available = list(model_factory.MODEL_REGISTRY.keys())
    return {
        "available_models": available,
        "trained_models": list(trained_models.keys()),
    }


@app.post("/models/train")
def train_models() -> Dict[str, Any]:
    """Train all enabled models on synthetic data.

    Returns:
        Dictionary with training results.
    """
    global trained_models

    from src.data.synthetic import SyntheticDataGenerator
    from src.preprocessing.cleaner import DataCleaner
    from src.preprocessing.transformer import DataTransformer

    # Generate synthetic data (sized appropriately for interactive API response)
    generator = SyntheticDataGenerator()
    datasets = generator.generate_all(n_records=1000)

    # Combine and prepare
    frames = []
    for name, df in datasets.items():
        df = df.copy()
        df["source_type"] = name
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    # Clean and transform
    cleaner = DataCleaner()
    transformer = DataTransformer()
    combined = cleaner.clean(combined)
    combined = transformer.transform(combined)

    # Prepare features
    exclude_cols = [
        "is_attack", "is_outlier", "risk_label", "source_type",
        "attack_type", "timestamp", "user_id", "host_id", "source_ip",
        "destination_host", "process_name", "parent_process", "location", "auth_method"
    ]
    feature_cols = [
        c for c in combined.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(combined[c])
    ]
    X = combined[feature_cols].fillna(0).values
    y = combined["is_attack"].astype(int).values

    # Train models
    trained_models = model_trainer.train_all(X, y)

    return {
        "trained_models": list(trained_models.keys()),
        "n_samples": len(X),
        "n_features": X.shape[1],
    }


# --- Database ---

@app.get("/events")
def get_events(
    limit: int = Query(50, ge=1, le=500),
    source_type: Optional[str] = None,
    risk_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Get recent security events.

    Args:
        limit: Maximum number of events.
        source_type: Filter by source type.
        risk_label: Filter by risk label.

    Returns:
        Dictionary with events list.
    """
    events = repository.get_events(
        limit=limit,
        source_type=source_type,
        risk_label=risk_label,
    )
    return {
        "events": [
            {
                "id": e.id,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                "source_type": e.source_type,
                "event_type": e.event_type,
                "risk_label": e.risk_label,
                "attack_type": e.attack_type,
            }
            for e in events
        ]
    }


@app.get("/risk-distribution")
def risk_distribution() -> Dict[str, Any]:
    """Get the distribution of risk levels.

    Returns:
        Dictionary with risk level distribution.
    """
    return {"distribution": repository.get_risk_score_distribution()}


@app.get("/dashboard")
def dashboard() -> Dict[str, Any]:
    """Get dashboard summary data.

    Returns:
        Dictionary with dashboard metrics.
    """
    events = repository.get_recent_events_with_risk(limit=50)
    distribution = repository.get_risk_score_distribution()

    return {
        "total_events": len(events),
        "risk_distribution": distribution,
        "recent_events": events,
        "models_loaded": len(trained_models),
        "rl_trained": len(rl_agent.q_table) > 0,
    }


# --- Ethics & ML Security ---

@app.get("/ethics/audit")
def audit_ethics() -> Dict[str, Any]:
    """Run an ethical AI, privacy, and adversarial security audit."""
    try:
        from src.pipeline import CyberRiskPipeline

        pipeline = CyberRiskPipeline()
        pipeline.run_data_pipeline(sample_per_source=1500)
        pipeline.run_training(optimize=False)
        pipeline.run_evaluation()
        pipeline.run_risk_fusion()
        results = pipeline.run_ethics_audit()
        return {"status": "success", "audit": results}
    except Exception as e:
        logger.error(f"Ethics audit failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# --- Multi-Agent RL ---

@app.post("/rl/multi-agent")
def analyze_multi_agent(state: Optional[List[float]] = None) -> Dict[str, Any]:
    """Analyze a security event using coordinated multi-agent specialist architecture."""
    try:
        from src.rl.multi_agent import MultiAgentSystem
        mas = MultiAgentSystem()
        if state is None or len(state) == 0:
            state_vec = np.array([0.85, 0.70, 0.60, 0.80, 0.75, 0.90])
        else:
            state_vec = np.asarray(state, dtype=float)

        res = mas.analyze_event(state_vec)
        return {
            "status": "success",
            "specialist_signals": res["signals"],
            "recommended_action": res["recommended_action"],
            "coordination": res["coordination"],
        }
    except Exception as e:
        logger.error(f"Multi-agent analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Database Seeding ---

@app.post("/database/seed")
def seed_database(count: int = Query(default=50, ge=10, le=500)) -> Dict[str, Any]:
    """Seed the database with synthetic events and risk intelligence."""
    try:
        pipeline = CyberRiskPipeline()
        saved = pipeline.seed_database(n_records=count)
        return {"status": "success", "seeded_events": saved}
    except Exception as e:
        logger.error(f"Database seeding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))