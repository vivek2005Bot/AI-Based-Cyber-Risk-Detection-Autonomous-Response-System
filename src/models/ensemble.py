"""Ensemble methods: voting, stacking, and bagging."""

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import BaggingClassifier, StackingClassifier, VotingClassifier

from src.models.supervised import SupervisedModelFactory
from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EnsembleBuilder:
    """Builds ensemble models from base estimators."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().get_section("ensemble")
        self.factory = SupervisedModelFactory()
        self.random_state = get_config().get("preprocessing.random_state", 42)

    def _get_base_estimators(self) -> List[tuple]:
        """Get base estimators from enabled models."""
        estimators: List[tuple] = []
        supervised_config = get_config().get_section("supervised_models")

        for name, model_config in supervised_config.items():
            if isinstance(model_config, dict) and model_config.get("enabled", False):
                try:
                    model = self.factory.create_model(name)
                    estimators.append((name, model))
                except Exception as e:
                    logger.warning(f"Could not create base estimator {name}: {e}")

        return estimators

    def build_voting(self) -> Optional[VotingClassifier]:
        """Build a voting ensemble classifier.

        Returns:
            VotingClassifier or None if not enough estimators.
        """
        if not self.config.get("enabled", True) or "voting" not in self.config.get("methods", []):
            return None

        estimators = self._get_base_estimators()
        if len(estimators) < 2:
            logger.warning("Not enough base estimators for voting ensemble.")
            return None

        voting_estimator_names = self.config.get("voting_estimators")
        if voting_estimator_names:
            estimators = [(n, m) for n, m in estimators if n in voting_estimator_names]
        else:
            fast_names = {"logistic_regression", "random_forest", "decision_tree", "naive_bayes", "knn"}
            fast_estimators = [(n, m) for n, m in estimators if n in fast_names]
            if len(fast_estimators) >= 2:
                estimators = fast_estimators

        voting_type = self.config.get("voting_type", "soft")
        logger.info(f"Building voting ensemble ({voting_type}) with {len(estimators)} estimators")
        return VotingClassifier(
            estimators=estimators,
            voting=voting_type,
            n_jobs=-1,
        )

    def build_stacking(self) -> Optional[StackingClassifier]:
        """Build a stacking ensemble classifier.

        Returns:
            StackingClassifier or None if not enough estimators.
        """
        if not self.config.get("enabled", True) or "stacking" not in self.config.get("methods", []):
            return None

        estimators = self._get_base_estimators()
        if len(estimators) < 2:
            logger.warning("Not enough base estimators for stacking ensemble.")
            return None

        stacking_estimator_names = self.config.get("stacking_estimators")
        if stacking_estimator_names:
            estimators = [(n, m) for n, m in estimators if n in stacking_estimator_names]
        else:
            # Exclude computationally expensive SVM and deep net from multi-fold cross-validation
            fast_names = {"logistic_regression", "random_forest", "decision_tree", "naive_bayes"}
            fast_estimators = [(n, m) for n, m in estimators if n in fast_names]
            if len(fast_estimators) >= 2:
                estimators = fast_estimators

        meta_learner_name = self.config.get("stacking_meta_learner", "logistic_regression")
        try:
            meta_learner = self.factory.create_model(meta_learner_name)
        except Exception as e:
            logger.warning(f"Could not create meta-learner {meta_learner_name}: {e}. Using LogisticRegression.")
            from sklearn.linear_model import LogisticRegression

            meta_learner = LogisticRegression(max_iter=1000)

        cv = self.config.get("stacking_cv", 3)
        logger.info(f"Building stacking ensemble with meta-learner {meta_learner_name} (cv={cv})")
        return StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=cv,
            n_jobs=-1,
        )

    def build_bagging(self, base_model_name: str = "decision_tree") -> Optional[BaggingClassifier]:
        """Build a bagging ensemble classifier.

        Args:
            base_model_name: Name of the base model to bag.

        Returns:
            BaggingClassifier or None if not enabled.
        """
        if not self.config.get("enabled", True) or "bagging" not in self.config.get("methods", []):
            return None

        try:
            base_model = self.factory.create_model(base_model_name)
        except Exception as e:
            logger.warning(f"Could not create base model {base_model_name}: {e}")
            return None

        logger.info(f"Building bagging ensemble with {base_model_name}")
        return BaggingClassifier(
            estimator=base_model,
            n_estimators=50,
            max_samples=0.8,
            max_features=0.8,
            bootstrap=True,
            bootstrap_features=False,
            n_jobs=-1,
            random_state=self.random_state,
        )

    def build_all(self) -> Dict[str, Any]:
        """Build all enabled ensemble methods.

        Returns:
            Dictionary of ensemble name to model.
        """
        ensembles: Dict[str, Any] = {}

        voting = self.build_voting()
        if voting is not None:
            ensembles["voting"] = voting

        stacking = self.build_stacking()
        if stacking is not None:
            ensembles["stacking"] = stacking

        bagging = self.build_bagging()
        if bagging is not None:
            ensembles["bagging"] = bagging

        return ensembles