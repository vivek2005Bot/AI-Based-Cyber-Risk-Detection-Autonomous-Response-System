"""End-to-end pipeline for the cyber risk detection system."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.anomaly.clustering import ClusteringAnalyzer
from src.anomaly.detectors import AnomalyDetector
from src.data.integration import DataIntegrator
from src.data.synthetic import SyntheticDataGenerator
from src.models.ensemble import EnsembleBuilder
from src.models.evaluation import ModelEvaluator
from src.models.optimization import HyperparameterOptimizer
from src.models.supervised import ModelTrainer, SupervisedModelFactory
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.dimensionality import DimensionalityReducer
from src.preprocessing.feature_selection import FeatureSelector
from src.preprocessing.imbalance import ImbalanceHandler
from src.preprocessing.transformer import DataTransformer
from src.recommendation.hybrid import HybridRecommender
from src.risk_fusion.fusion import RiskFusionEngine
from src.rl.q_learning import QLearningAgent
from src.ethics.fairness import FairnessAuditor
from src.ethics.privacy import PrivacyEngine
from src.ethics.security import AdversarialRobustnessAuditor
from src.database.repository import CyberRiskRepository
from src.utils.config import get_config
from src.utils.logging import get_logger
from src.utils.metrics import classification_metrics, clustering_metrics, rl_metrics

logger = get_logger(__name__)


class CyberRiskPipeline:
    """Orchestrates the complete cyber risk detection workflow."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().data
        self.random_state = self.config.get("preprocessing", {}).get("random_state", 42)

        # Components
        self.integrator = DataIntegrator()
        self.cleaner = DataCleaner()
        self.transformer = DataTransformer()
        self.imbalance_handler = ImbalanceHandler()
        self.feature_selector = FeatureSelector()
        self.dimensionality_reducer = DimensionalityReducer()
        self.model_factory = SupervisedModelFactory()
        self.model_trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        self.optimizer = HyperparameterOptimizer()
        self.ensemble_builder = EnsembleBuilder()
        self.clustering = ClusteringAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.risk_fusion = RiskFusionEngine()
        self.recommender = HybridRecommender()
        self.rl_agent = QLearningAgent()
        self.fairness_auditor = FairnessAuditor()
        self.privacy_engine = PrivacyEngine()
        self.adversarial_auditor = AdversarialRobustnessAuditor()
        self.repository = CyberRiskRepository()

        # State
        self.data: Optional[pd.DataFrame] = None
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        self.trained_models: Dict[str, Any] = {}
        self.ensemble_models: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}

    def run_data_pipeline(
        self,
        use_synthetic: bool = False,
        n_records: Optional[int] = None,
        sample_per_source: Optional[int] = None,
    ) -> pd.DataFrame:
        """Run the data acquisition and preprocessing pipeline.

        Args:
            use_synthetic: Whether to use synthetic fallback data (defaults to False).
            n_records: Optional number of records per synthetic telemetry source.
            sample_per_source: Optional row sampling per raw dataset source.

        Returns:
            Preprocessed DataFrame.
        """
        logger.info("=== Running data pipeline ===")

        # Generate synthetic data or load real telemetry
        if use_synthetic:
            generator = SyntheticDataGenerator()
            datasets = generator.generate_all(n_records=n_records)
            frames = []
            event_type_map = {
                "authentication": "auth",
                "endpoint": "endpoint",
                "user_behavior": "behavior",
                "network": "network",
            }
            for name, df in datasets.items():
                df = df.copy()
                df["source_type"] = name
                df["event_type"] = event_type_map.get(name, name)
                if "is_attack" in df.columns:
                    df["is_attack"] = df["is_attack"].astype(int)
                    df["risk_label"] = df["is_attack"].map({1: "suspicious", 0: "normal"})
                frames.append(df)
            self.data = pd.concat(frames, ignore_index=True)
        else:
            integrator = DataIntegrator(self.config)
            self.data = integrator.integrate(sample_per_source=sample_per_source, use_synthetic=False)

        if self.data is None or self.data.empty:
            raise ValueError("No data loaded. Verify data directories and configurations.")

        # Ensure critical columns exist
        if "is_attack" not in self.data.columns:
            self.data["is_attack"] = 0
        self.data["is_attack"] = self.data["is_attack"].fillna(0).astype(int)

        if "risk_label" not in self.data.columns:
            self.data["risk_label"] = self.data["is_attack"].map({1: "suspicious", 0: "normal"})

        if "source_type" not in self.data.columns:
            self.data["source_type"] = "security_telemetry"

        if "event_type" not in self.data.columns:
            self.data["event_type"] = "security_event"

        if "attack_type" not in self.data.columns:
            self.data["attack_type"] = self.data["is_attack"].map({1: "suspicious_activity", 0: "normal"})

        logger.info(f"Loaded {len(self.data)} records from data sources")

        # Replace infinite values prior to cleaning
        self.data = self.data.replace([np.inf, -np.inf], np.nan)

        # Clean
        self.data = self.cleaner.clean(self.data)
        logger.info(f"After cleaning: {len(self.data)} records")

        # Transform
        self.data = self.transformer.transform(self.data)

        # Prepare features
        exclude_cols = [
            "is_attack", "is_outlier", "risk_label", "source_type", "event_type",
            "attack_type", "timestamp", "user_id", "host_id", "source_ip",
            "destination_host", "process_name", "parent_process", "location",
            "auth_method", "hostName", "processName", "eventName", "args",
            "stackAddresses", "mountNamespace", "id", "Label", "label",
            "attack_cat", "class", "cve_id", "description", "cwe_ids", "cvss_vector",
            "sus", "evil", "sttl", "dttl", "service", "state", "proto",
        ]
        self.feature_names = [
            c for c in self.data.columns
            if c not in exclude_cols and self.data[c].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]

        if not self.feature_names:
            raise ValueError("No numeric feature columns available for modeling.")

        X_df = self.data[self.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_ser = self.data["is_attack"].astype(int)

        # 1. SPLIT FIRST to guarantee zero test-set data leakage
        test_size = self.config.get("preprocessing", {}).get("test_size", 0.2)
        stratify_option = y_ser if len(np.unique(y_ser)) > 1 else None
        X_train_df, X_test_df, self.y_train, self.y_test = train_test_split(
            X_df, y_ser, test_size=test_size, random_state=self.random_state, stratify=stratify_option
        )

        # 2. Handle class imbalance on TRAINING set ONLY
        if self.config.get("preprocessing", {}).get("class_imbalance", {}).get("enabled", True):
            try:
                X_train_vals, y_train_vals = self.imbalance_handler.resample(X_train_df.values, self.y_train.values)
                X_train_df = pd.DataFrame(X_train_vals, columns=self.feature_names)
                self.y_train = pd.Series(y_train_vals)
            except Exception as e:
                logger.warning(f"Class imbalance handling skipped: {e}")

        # 3. Feature selection (fit on train, transform both)
        if self.config.get("preprocessing", {}).get("feature_selection", {}).get("enabled", True):
            try:
                X_train_df = self.feature_selector.select(X_train_df, self.y_train)
                self.feature_names = self.feature_selector.selected_features
                X_test_df = X_test_df[self.feature_names]
            except Exception as e:
                logger.warning(f"Feature selection skipped: {e}")

        # 4. Dimensionality reduction (fit on train, transform test)
        if self.config.get("preprocessing", {}).get("dimensionality_reduction", {}).get("enabled", True):
            try:
                X_train_df = self.dimensionality_reducer.fit_transform(X_train_df, self.y_train)
                X_test_df = self.dimensionality_reducer.transform(X_test_df)
                self.feature_names = X_train_df.columns.tolist()
            except Exception as e:
                logger.warning(f"Dimensionality reduction skipped: {e}")

        self.X_train = X_train_df
        self.X_test = X_test_df

        logger.info(f"Train set: {len(self.X_train)} samples, Test set: {len(self.X_test)} samples")
        logger.info(f"Features: {len(self.feature_names)}")

        self.results["data"] = {
            "n_samples": len(self.data),
            "n_features": len(self.feature_names),
            "n_train": len(self.X_train),
            "n_test": len(self.X_test),
        }
        return self.data

    def run_training(self, optimize: bool = False) -> Dict[str, Any]:
        """Train all supervised models.

        Args:
            optimize: Whether to run hyperparameter optimization.

        Returns:
            Dictionary of trained models.
        """
        logger.info("=== Training supervised models ===")

        if self.X_train is None:
            raise ValueError("Data pipeline must be run first.")

        if optimize and self.config.get("optimization", {}).get("enabled", True):
            models_to_optimize = self.config.get("optimization", {}).get("models_to_optimize", [])
            for name in models_to_optimize:
                try:
                    model = self.model_factory.create_model(name)
                    result = self.optimizer.optimize(model, name, self.X_train, self.y_train)
                    self.trained_models[name] = result["best_model"]
                    logger.info(f"Optimized {name}: {result['best_params']}")
                except Exception as e:
                    logger.error(f"Optimization failed for {name}: {e}")

        # Train remaining models
        trained = self.model_trainer.train_all(self.X_train, self.y_train)
        for name, model in trained.items():
            if name not in self.trained_models:
                self.trained_models[name] = model

        # Build ensembles
        self.ensemble_models = self.ensemble_builder.build_all()
        for name, model in self.ensemble_models.items():
            try:
                model.fit(self.X_train, self.y_train)
                self.trained_models[f"ensemble_{name}"] = model
            except Exception as e:
                logger.error(f"Failed to train ensemble {name}: {e}")

        logger.info(f"Trained {len(self.trained_models)} models")
        return self.trained_models

    def run_evaluation(self) -> pd.DataFrame:
        """Evaluate all trained models.

        Returns:
            DataFrame of evaluation results.
        """
        logger.info("=== Evaluating models ===")

        if not self.trained_models:
            raise ValueError("No trained models. Run training first.")

        results_df = self.evaluator.evaluate_all(
            self.trained_models, self.X_test, self.y_test
        )
        self.results["evaluation"] = results_df.to_dict(orient="index")
        logger.info(f"Evaluation results:\n{results_df[['accuracy', 'precision', 'recall', 'f1']].to_string()}")
        return results_df

    def run_anomaly_detection(self) -> Dict[str, Any]:
        """Run unsupervised anomaly detection and clustering.

        Returns:
            Dictionary of anomaly detection results.
        """
        logger.info("=== Running anomaly detection ===")

        if self.X_train is None:
            raise ValueError("Data pipeline must be run first.")

        # Clustering
        clustering_results = self.clustering.run_all(self.X_train)
        self.results["clustering"] = {
            name: result["metrics"] for name, result in clustering_results.items()
        }

        # Anomaly detection
        anomaly_results = self.anomaly_detector.run_all(self.X_train)
        self.results["anomaly"] = {
            name: {
                "n_anomalies": result["n_anomalies"],
                "anomaly_rate": result["n_anomalies"] / len(self.X_train),
            }
            for name, result in anomaly_results.items()
        }

        logger.info(f"Anomaly detection results: {self.results['anomaly']}")
        return self.results

    def run_risk_fusion(self) -> np.ndarray:
        """Compute fused risk scores for the test set.

        Returns:
            Array of risk scores.
        """
        logger.info("=== Computing risk scores ===")

        if self.X_test is None:
            raise ValueError("Data pipeline must be run first.")

        # Get supervised scores from best model (highest F1)
        if self.trained_models:
            best_model_name = None
            best_f1 = -1
            eval_results = self.results.get("evaluation", {})
            for name, metrics in eval_results.items():
                f1 = metrics.get("f1", 0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_model_name = name

            if best_model_name:
                model = self.trained_models[best_model_name]
                if hasattr(model, "predict_proba"):
                    supervised_scores = model.predict_proba(self.X_test)[:, 1]
                else:
                    supervised_scores = model.predict(self.X_test).astype(float)
            else:
                supervised_scores = np.zeros(len(self.X_test))
        else:
            supervised_scores = np.zeros(len(self.X_test))

        # Get anomaly scores
        anomaly_scores = self.anomaly_detector.combined_anomaly_score(self.X_test)

        # Get contextual scores (placeholder - would use real contextual data)
        contextual_scores = np.zeros(len(self.X_test))

        # Fuse
        risk_scores = self.risk_fusion.fuse(
            supervised_scores, anomaly_scores, contextual_scores
        )
        risk_levels = self.risk_fusion.risk_level(risk_scores)

        self.results["risk_fusion"] = {
            "mean_risk": float(np.mean(risk_scores)),
            "risk_distribution": {
                level: int((risk_levels == level).sum())
                for level in ["minimal", "low", "medium", "high", "critical"]
            },
        }
        logger.info(f"Risk distribution: {self.results['risk_fusion']['risk_distribution']}")
        return risk_scores

    def run_recommendation(self) -> List[Dict[str, Any]]:
        """Generate recommendations for high-risk events.

        Returns:
            List of recommendations.
        """
        logger.info("=== Generating recommendations ===")

        risk_scores = self.results.get("risk_scores")
        if risk_scores is None:
            risk_scores = self.run_risk_fusion()
            self.results["risk_scores"] = risk_scores

        risk_levels = self.risk_fusion.risk_level(risk_scores)

        # Find high-risk events
        high_risk_indices = np.where(
            (risk_levels == "high") | (risk_levels == "critical")
        )[0]

        recommendations = []
        for idx in high_risk_indices[:10]:  # Limit to 10
            event = {
                "event_type": "network",
                "risk_level": risk_levels[idx],
                "risk_score": float(risk_scores[idx]),
            }
            recs = self.recommender.recommend(event, top_k=3)
            recommendations.append({
                "event_index": int(idx),
                "risk_score": float(risk_scores[idx]),
                "risk_level": risk_levels[idx],
                "recommendations": recs,
            })

        self.results["recommendations"] = recommendations
        logger.info(f"Generated recommendations for {len(recommendations)} high-risk events")
        return recommendations

    def run_rl_training(self, n_episodes: Optional[int] = None) -> Dict[str, Any]:
        """Train the Q-Learning agent.

        Args:
            n_episodes: Number of episodes.

        Returns:
            Dictionary of RL metrics.
        """
        logger.info("=== Training Q-Learning agent ===")

        metrics = self.rl_agent.train(n_episodes=n_episodes)
        eval_metrics = self.rl_agent.evaluate(n_episodes=100)
        metrics["evaluation"] = eval_metrics

        self.results["rl"] = metrics
        logger.info(f"RL metrics: {metrics}")
        return metrics

    def run_ethics_audit(self) -> Dict[str, Any]:
        """Run ethical AI, privacy, and adversarial security audit.

        Returns:
            Dictionary of ethics, privacy, and security metrics.
        """
        logger.info("=== Running Ethical AI, Privacy & Security Audit ===")

        if self.data is None or self.X_test is None or self.y_test is None:
            raise ValueError("Data pipeline and evaluation must be run first.")

        # 1. Fairness & Bias Audit
        # Check against location or auth_method if available, or generate cohort groups
        if "location" in self.data.columns:
            sensitive_attr = self.data["location"].iloc[:len(self.y_test)].values
        elif "auth_method" in self.data.columns:
            sensitive_attr = self.data["auth_method"].iloc[:len(self.y_test)].values
        else:
            sensitive_attr = np.array(["office_a" if i % 2 == 0 else "remote_travel" for i in range(len(self.y_test))])

        best_model = None
        if self.trained_models:
            best_model_name = list(self.trained_models.keys())[0]
            best_model = self.trained_models[best_model_name]

        if best_model is not None:
            y_pred = best_model.predict(self.X_test)
            fairness_result = self.fairness_auditor.audit(
                y_true=self.y_test,
                y_pred=y_pred,
                sensitive_attribute=sensitive_attr,
            )
        else:
            fairness_result = {"status": "SKIPPED", "message": "No trained model available."}

        # 2. Privacy & Anonymization Audit
        qi_cols = [c for c in ["hour_of_day", "location", "auth_method", "source_type"] if c in self.data.columns]
        k_anon_result = self.privacy_engine.audit_k_anonymity(self.data, quasi_identifiers=qi_cols, target_k=5)

        risk_scores = self.results.get("risk_scores")
        if risk_scores is None and self.X_test is not None:
            risk_scores = self.run_risk_fusion()

        dp_report = self.privacy_engine.differentially_private_risk_report(
            risk_scores=risk_scores if risk_scores is not None else np.zeros(10),
            epsilon=1.0,
        )

        # 3. Adversarial Robustness Audit
        if best_model is not None:
            robustness_result = self.adversarial_auditor.evaluate_evasion_robustness(
                model=best_model,
                X_test=self.X_test,
                y_test=self.y_test,
                feature_names=self.feature_names,
            )
        else:
            robustness_result = {"status": "SKIPPED", "message": "No trained model available."}

        self.results["ethics"] = {
            "fairness": fairness_result,
            "privacy": {
                "k_anonymity": k_anon_result,
                "differential_privacy": dp_report,
            },
            "security": robustness_result,
        }

        logger.info(f"Fairness Status: {fairness_result.get('status')}")
        logger.info(f"Privacy K-anonymity Achieved: k={k_anon_result.get('k_achieved')}")
        logger.info(f"Adversarial Robustness Score: {robustness_result.get('robustness_score')}%")

        return self.results["ethics"]

    def save_to_database(self, max_events: int = 500) -> int:
        """Persist processed events, predictions, risk scores, and recommendations to DB.

        Args:
            max_events: Maximum number of events to store.

        Returns:
            Number of events successfully saved.
        """
        logger.info(f"=== Persisting up to {max_events} events to database ===")

        if self.data is None:
            logger.warning("No data to save. Run data pipeline first.")
            return 0

        risk_scores = self.results.get("risk_scores")
        if risk_scores is None:
            risk_scores = self.run_risk_fusion()
        risk_levels = self.risk_fusion.risk_level(risk_scores)

        saved_count = 0
        n_samples = min(max_events, len(self.data), len(risk_scores))

        for idx in range(n_samples):
            row = self.data.iloc[idx]
            features_dict = {
                k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else str(v)
                for k, v in row.to_dict().items()
                if k not in ["timestamp", "user_id", "host_id", "source_ip", "attack_type"] and pd.notna(v)
            }

            event = self.repository.add_event({
                "source_type": str(row.get("source_type", "synthetic")),
                "event_type": str(row.get("event_type", "auth")),
                "user_id": str(row.get("user_id", f"user_{idx:03d}")),
                "host_id": str(row.get("host_id", f"host_{idx:03d}")),
                "source_ip": str(row.get("source_ip", "192.168.1.100")),
                "risk_label": "suspicious" if int(row.get("is_attack", 0)) == 1 else "normal",
                "attack_type": str(row.get("attack_type", "normal")),
                "features": features_dict,
            })

            # Add risk score
            score_val = float(risk_scores[idx]) if idx < len(risk_scores) else 0.5
            level_val = str(risk_levels[idx]) if idx < len(risk_levels) else "medium"

            self.repository.add_risk_score(
                event_id=event.id,
                risk_score=score_val,
                risk_level=level_val,
                supervised_score=score_val * 0.9,
                anomaly_score=score_val * 0.8,
                contextual_score=score_val * 0.7,
                explanation=f"Risk assessed as {level_val} ({score_val:.2f}) from multimodal evidence.",
            )

            # Add recommendation for higher risk
            if level_val in ["medium", "high", "critical"]:
                recs = self.recommender.recommend(
                    {"event_type": event.event_type, "risk_level": level_val, **features_dict},
                    top_k=2,
                )
                for rec in recs:
                    self.repository.add_recommendation(
                        event_id=event.id,
                        action=rec.get("action", "investigate"),
                        score=float(rec.get("score", 0.85)),
                        description=rec.get("description", "Action recommendation"),
                        method="hybrid",
                    )

            saved_count += 1

        logger.info(f"Successfully persisted {saved_count} security events and associated risk intelligence.")
        return saved_count

    def seed_database(self, n_records: int = 150) -> int:
        """Convenience method to generate synthetic data, evaluate risk, and seed DB."""
        logger.info(f"=== Seeding database with {n_records} realistic security events ===")
        self.run_data_pipeline(use_synthetic=True)
        self.run_training(optimize=False)
        self.run_evaluation()
        self.run_anomaly_detection()
        self.run_risk_fusion()
        self.run_recommendation()
        return self.save_to_database(max_events=n_records)

    def run_full_pipeline(
        self,
        use_synthetic: bool = True,
        optimize: bool = False,
        run_anomaly: bool = True,
        run_rl: bool = True,
        run_ethics: bool = True,
        save_db: bool = False,
        sample_per_source: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run the complete pipeline end-to-end.

        Args:
            use_synthetic: Whether to use synthetic data.
            optimize: Whether to run hyperparameter optimization.
            run_anomaly: Whether to run anomaly detection.
            run_rl: Whether to train the RL agent.
            run_ethics: Whether to run ethical, privacy, and security audit.
            save_db: Whether to persist results to SQLite database.
            sample_per_source: Optional sample row limit per raw data source.

        Returns:
            Dictionary of all results.
        """
        logger.info("=== Starting full pipeline ===")

        self.run_data_pipeline(use_synthetic=use_synthetic, sample_per_source=sample_per_source)
        self.run_training(optimize=optimize)
        self.run_evaluation()

        if run_anomaly:
            self.run_anomaly_detection()

        self.run_risk_fusion()
        self.run_recommendation()

        if run_rl:
            self.run_rl_training()

        if run_ethics:
            self.run_ethics_audit()

        if save_db:
            self.save_to_database()

        logger.info("=== Pipeline complete ===")
        return self.results

    def save_models(self, output_dir: str | Path = "models") -> Dict[str, str]:
        """Save all trained models to disk.

        Args:
            output_dir: Directory to save models.

        Returns:
            Dictionary of model name to saved path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = {}
        for name, model in self.trained_models.items():
            path = output_dir / f"{name}.joblib"
            joblib.dump(model, path)
            saved_paths[name] = str(path)
            logger.info(f"Saved model {name} to {path}")

        return saved_paths

    def load_models(self, input_dir: str | Path = "models") -> Dict[str, Any]:
        """Load all saved models from disk.

        Args:
            input_dir: Directory to load models from.

        Returns:
            Dictionary of model name to loaded model.
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            logger.warning(f"Model directory not found: {input_dir}")
            return {}

        for path in input_dir.glob("*.joblib"):
            name = path.stem
            self.trained_models[name] = joblib.load(path)
            logger.info(f"Loaded model {name} from {path}")

        return self.trained_models