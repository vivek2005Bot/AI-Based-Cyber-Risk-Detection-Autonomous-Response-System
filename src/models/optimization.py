"""Hyperparameter optimization using Grid Search, Random Search, and Bayesian Optimization."""

from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import optuna

    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False


class HyperparameterOptimizer:
    """Optimizes model hyperparameters using various search strategies."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().get_section("optimization")
        self.method = self.config.get("method", "bayesian")
        self.n_trials = self.config.get("n_trials", 50)
        self.timeout = self.config.get("timeout", 3600)
        self.cv_folds = self.config.get("cv_folds", 5)
        self.scoring = self.config.get("scoring", "f1_weighted")
        self.random_state = get_config().get("preprocessing.random_state", 42)

    def _get_param_grid(self, model_name: str) -> Dict[str, List[Any]]:
        """Get hyperparameter search space for a model.

        Args:
            model_name: Name of the model.

        Returns:
            Dictionary of parameter names to search values.
        """
        grids: Dict[str, Dict[str, List[Any]]] = {
            "logistic_regression": {
                "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "solver": ["lbfgs", "liblinear", "saga"],
                "max_iter": [500, 1000, 2000],
            },
            "knn": {
                "n_neighbors": [3, 5, 7, 9, 11, 15],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan", "minkowski"],
            },
            "svm": {
                "C": [0.1, 1.0, 10.0, 100.0],
                "gamma": ["scale", "auto", 0.01, 0.1, 1.0],
                "kernel": ["rbf", "linear", "poly"],
            },
            "decision_tree": {
                "max_depth": [5, 10, 15, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "criterion": ["gini", "entropy"],
            },
            "random_forest": {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [10, 15, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
            },
            "neural_network": {
                "hidden_layer_sizes": [(64,), (128,), (128, 64), (128, 64, 32)],
                "activation": ["relu", "tanh"],
                "alpha": [0.0001, 0.001, 0.01],
                "learning_rate": ["constant", "adaptive"],
            },
        }
        return grids.get(model_name, {})

    def _get_optuna_params(self, trial: Any, model_name: str) -> Dict[str, Any]:
        """Get Optuna parameter suggestions for a model.

        Args:
            trial: Optuna trial object.
            model_name: Name of the model.

        Returns:
            Dictionary of parameter values.
        """
        params: Dict[str, Any] = {}

        if model_name == "logistic_regression":
            params["C"] = trial.suggest_float("C", 0.01, 100.0, log=True)
            params["solver"] = trial.suggest_categorical("solver", ["lbfgs", "liblinear", "saga"])
            params["max_iter"] = trial.suggest_int("max_iter", 500, 2000, step=500)
        elif model_name == "knn":
            params["n_neighbors"] = trial.suggest_int("n_neighbors", 3, 15)
            params["weights"] = trial.suggest_categorical("weights", ["uniform", "distance"])
            params["metric"] = trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"])
        elif model_name == "svm":
            params["C"] = trial.suggest_float("C", 0.1, 100.0, log=True)
            params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
            params["kernel"] = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])
            params["probability"] = True
        elif model_name == "decision_tree":
            params["max_depth"] = trial.suggest_int("max_depth", 5, 20)
            params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 10)
            params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 4)
            params["criterion"] = trial.suggest_categorical("criterion", ["gini", "entropy"])
        elif model_name == "random_forest":
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 300, step=50)
            params["max_depth"] = trial.suggest_int("max_depth", 10, 20)
            params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 10)
            params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 4)
            params["max_features"] = trial.suggest_categorical("max_features", ["sqrt", "log2"])
        elif model_name == "neural_network":
            n_layers = trial.suggest_int("n_layers", 1, 3)
            hidden = tuple(trial.suggest_int(f"hidden_{i}", 32, 256, step=32) for i in range(n_layers))
            params["hidden_layer_sizes"] = hidden
            params["activation"] = trial.suggest_categorical("activation", ["relu", "tanh"])
            params["alpha"] = trial.suggest_float("alpha", 0.0001, 0.01, log=True)
            params["learning_rate"] = trial.suggest_categorical("learning_rate", ["constant", "adaptive"])

        return params

    def optimize(
        self,
        model: Any,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Dict[str, Any]:
        """Optimize model hyperparameters.

        Args:
            model: Base model instance.
            model_name: Name of the model.
            X_train: Training features.
            y_train: Training labels.

        Returns:
            Dictionary with 'best_model', 'best_params', and 'best_score'.
        """
        if self.method == "grid":
            return self._grid_search(model, model_name, X_train, y_train)
        elif self.method == "random":
            return self._random_search(model, model_name, X_train, y_train)
        elif self.method == "bayesian":
            return self._bayesian_optimization(model, model_name, X_train, y_train)
        else:
            logger.warning(f"Unknown optimization method: {self.method}. Using grid search.")
            return self._grid_search(model, model_name, X_train, y_train)

    def _grid_search(
        self, model: Any, model_name: str, X_train: np.ndarray, y_train: np.ndarray
    ) -> Dict[str, Any]:
        """Perform grid search optimization."""
        param_grid = self._get_param_grid(model_name)
        if not param_grid:
            logger.warning(f"No parameter grid defined for {model_name}. Returning base model.")
            return {"best_model": model, "best_params": {}, "best_score": None}

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        search = GridSearchCV(
            model,
            param_grid,
            cv=cv,
            scoring=self.scoring,
            n_jobs=-1,
            verbose=1,
        )
        logger.info(f"Running grid search for {model_name}...")
        search.fit(X_train, y_train)
        logger.info(f"Best score for {model_name}: {search.best_score_:.4f}")
        return {
            "best_model": search.best_estimator_,
            "best_params": search.best_params_,
            "best_score": search.best_score_,
        }

    def _random_search(
        self, model: Any, model_name: str, X_train: np.ndarray, y_train: np.ndarray
    ) -> Dict[str, Any]:
        """Perform random search optimization."""
        param_grid = self._get_param_grid(model_name)
        if not param_grid:
            logger.warning(f"No parameter grid defined for {model_name}. Returning base model.")
            return {"best_model": model, "best_params": {}, "best_score": None}

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=self.n_trials,
            cv=cv,
            scoring=self.scoring,
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1,
        )
        logger.info(f"Running random search for {model_name}...")
        search.fit(X_train, y_train)
        logger.info(f"Best score for {model_name}: {search.best_score_:.4f}")
        return {
            "best_model": search.best_estimator_,
            "best_params": search.best_params_,
            "best_score": search.best_score_,
        }

    def _bayesian_optimization(
        self, model: Any, model_name: str, X_train: np.ndarray, y_train: np.ndarray
    ) -> Dict[str, Any]:
        """Perform Bayesian optimization using Optuna."""
        if not _HAS_OPTUNA:
            logger.warning("Optuna not installed. Falling back to random search.")
            return self._random_search(model, model_name, X_train, y_train)

        from sklearn.model_selection import cross_val_score

        def objective(trial: Any) -> float:
            params = self._get_optuna_params(trial, model_name)
            trial_model = model.__class__(**params)
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(trial_model, X_train, y_train, cv=cv, scoring=self.scoring, n_jobs=-1)
            return float(scores.mean())

        study = optuna.create_study(direction="maximize")
        logger.info(f"Running Bayesian optimization for {model_name} with {self.n_trials} trials...")
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        best_params = study.best_params
        # Convert Optuna params to model constructor params
        if model_name == "neural_network":
            hidden = tuple(best_params.pop(f"hidden_{i}") for i in range(best_params.pop("n_layers", 1)))
            best_params["hidden_layer_sizes"] = hidden

        best_model = model.__class__(**best_params)
        best_model.fit(X_train, y_train)

        logger.info(f"Best score for {model_name}: {study.best_value:.4f}")
        return {
            "best_model": best_model,
            "best_params": best_params,
            "best_score": study.best_value,
        }