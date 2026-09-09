"""Tests for the end-to-end cyber risk pipeline."""

import pytest
from src.pipeline import CyberRiskPipeline


def test_pipeline_end_to_end():
    """Test full pipeline workflow including data, models, anomaly, fusion, and ethics."""
    pipeline = CyberRiskPipeline()

    # Run data pipeline with synthetic data
    df = pipeline.run_data_pipeline(use_synthetic=True)
    assert df is not None
    assert len(df) > 0
    assert pipeline.X_train is not None
    assert pipeline.y_train is not None

    # Fast test slice
    pipeline.X_train = pipeline.X_train[:400]
    pipeline.y_train = pipeline.y_train[:400]
    pipeline.X_test = pipeline.X_test[:100]
    pipeline.y_test = pipeline.y_test[:100]

    # Run supervised training & evaluation
    trained = pipeline.run_training(optimize=False)
    assert len(trained) > 0

    eval_df = pipeline.run_evaluation()
    assert eval_df is not None
    assert "f1" in eval_df.columns
    assert "accuracy" in eval_df.columns

    # Run anomaly detection
    anom_results = pipeline.run_anomaly_detection()
    assert "clustering" in anom_results
    assert "anomaly" in anom_results

    # Run risk fusion & recommendation
    risk_scores = pipeline.run_risk_fusion()
    assert len(risk_scores) == len(pipeline.X_test)
    assert 0.0 <= risk_scores.mean() <= 1.0

    recs = pipeline.run_recommendation()
    assert isinstance(recs, list)

    # Run ethics audit
    ethics_res = pipeline.run_ethics_audit()
    assert "fairness" in ethics_res
    assert "privacy" in ethics_res
    assert "security" in ethics_res
