"""CLI entry point for the AI-Based Cyber Risk Detection System."""

import argparse
import sys
from pathlib import Path

from src.utils.logging import setup_logging

logger = setup_logging()


def cmd_generate_synthetic(args: argparse.Namespace) -> None:
    """Generate synthetic security data."""
    from src.data.synthetic import SyntheticDataGenerator

    generator = SyntheticDataGenerator()
    paths = generator.save_all(args.output)
    for name, path in paths.items():
        print(f"Generated {name}: {path}")


def cmd_run_pipeline(args: argparse.Namespace) -> None:
    """Run the full pipeline."""
    from src.pipeline import CyberRiskPipeline

    pipeline = CyberRiskPipeline()
    results = pipeline.run_full_pipeline(
        use_synthetic=args.synthetic,
        optimize=args.optimize,
        run_anomaly=not args.no_anomaly,
        run_rl=not args.no_rl,
        sample_per_source=getattr(args, "sample_size", None),
    )

    if args.save_models:
        paths = pipeline.save_models()
        print(f"Saved {len(paths)} models")

    print("\n=== Pipeline Results ===")
    print(f"Data: {results.get('data', {})}")
    print(f"Evaluation: {results.get('evaluation', {})}")
    print(f"Anomaly: {results.get('anomaly', {})}")
    print(f"Risk Fusion: {results.get('risk_fusion', {})}")
    print(f"RL: {results.get('rl', {})}")


def cmd_train(args: argparse.Namespace) -> None:
    """Train models only on real datasets."""
    from src.pipeline import CyberRiskPipeline

    pipeline = CyberRiskPipeline()
    pipeline.run_data_pipeline(
        use_synthetic=args.synthetic,
        sample_per_source=getattr(args, "sample_size", None),
    )
    pipeline.run_training(optimize=args.optimize)
    results = pipeline.run_evaluation()

    print("\n=== Model Comparison ===")
    print(results[["accuracy", "precision", "recall", "f1"]].to_string())

    if args.save_models:
        paths = pipeline.save_models()
        print(f"\nSaved {len(paths)} models")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate saved models."""
    from src.pipeline import CyberRiskPipeline

    pipeline = CyberRiskPipeline()
    pipeline.load_models(args.models_dir)
    if not pipeline.trained_models:
        print("No models found. Train models first.")
        sys.exit(1)

    pipeline.run_data_pipeline(use_synthetic=True)
    results = pipeline.run_evaluation()
    print(results[["accuracy", "precision", "recall", "f1"]].to_string())


def cmd_api(args: argparse.Namespace) -> None:
    """Start the FastAPI server."""
    import uvicorn

    from src.utils.config import get_config

    config = get_config()
    host = args.host or config.get("api.host", "0.0.0.0")
    port = args.port or config.get("api.port", 8001)

    print(f"Starting API server at http://{host}:{port}")
    uvicorn.run("src.api.main:app", host=host, port=port, reload=args.reload)


def cmd_rl_train(args: argparse.Namespace) -> None:
    """Train the Q-Learning agent."""
    from src.rl.q_learning import QLearningAgent

    agent = QLearningAgent()
    metrics = agent.train(n_episodes=args.episodes)
    eval_metrics = agent.evaluate(n_episodes=args.eval_episodes)

    print("\n=== RL Training Metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\n=== RL Evaluation Metrics ===")
    for key, value in eval_metrics.items():
        print(f"{key}: {value}")

    if args.save:
        agent.save(args.save)


def cmd_anomaly(args: argparse.Namespace) -> None:
    """Run anomaly detection."""
    from src.pipeline import CyberRiskPipeline

    pipeline = CyberRiskPipeline()
    pipeline.run_data_pipeline(use_synthetic=True)
    results = pipeline.run_anomaly_detection()

    print("\n=== Clustering Metrics ===")
    for name, metrics in results.get("clustering", {}).items():
        print(f"{name}: {metrics}")

    print("\n=== Anomaly Detection ===")
    for name, metrics in results.get("anomaly", {}).items():
        print(f"{name}: {metrics}")


def cmd_risk(args: argparse.Namespace) -> None:
    """Compute risk scores."""
    from src.pipeline import CyberRiskPipeline

    pipeline = CyberRiskPipeline()
    pipeline.run_data_pipeline(use_synthetic=True)
    pipeline.run_training()
    pipeline.run_evaluation()
    pipeline.run_anomaly_detection()
    risk_scores = pipeline.run_risk_fusion()

    print("\n=== Risk Fusion Results ===")
    print(f"Mean risk score: {pipeline.results['risk_fusion']['mean_risk']:.4f}")
    print(f"Risk distribution: {pipeline.results['risk_fusion']['risk_distribution']}")


def cmd_ethics(args: argparse.Namespace) -> None:
    """Run ethical AI, privacy, and adversarial security audit."""
    from src.pipeline import CyberRiskPipeline

    pipeline = CyberRiskPipeline()
    pipeline.run_data_pipeline(use_synthetic=True)
    pipeline.run_training()
    pipeline.run_evaluation()
    pipeline.run_risk_fusion()
    ethics_results = pipeline.run_ethics_audit()

    print("\n=== Ethical AI, Privacy & Security Audit ===")
    print(f"Fairness Status: {ethics_results['fairness'].get('status')}")
    print(f"Fairness Summary: {ethics_results['fairness'].get('summary')}")
    print(f"Privacy K-Anonymity Achieved: k={ethics_results['privacy']['k_anonymity'].get('k_achieved')}")
    print(f"Differential Privacy Report: {ethics_results['privacy']['differential_privacy']}")
    print(f"Adversarial Robustness Score: {ethics_results['security'].get('robustness_score')}% ({ethics_results['security'].get('status')})")


def cmd_multi_agent(args: argparse.Namespace) -> None:
    """Run multi-agent coordinated response analysis."""
    import numpy as np
    from src.rl.multi_agent import MultiAgentSystem

    mas = MultiAgentSystem()
    print("\n=== Multi-Agent Coordinated Response Analysis ===")
    sim_state = np.array([0.85, 0.70, 0.60, 0.80, 0.75, 0.90])
    res = mas.analyze_event(sim_state)
    for sig in res["signals"]:
        print(f"Specialist '{sig['agent']}' ({sig['focus']}): Threat Signal={sig['signal']:.3f}, Confidence={sig['confidence']:.2f}")
    print(f"==> Central Decision Agent Selected Action: {res['recommended_action'].upper()}")


def cmd_streamlit(args: argparse.Namespace) -> None:
    """Launch the Streamlit Web Application."""
    import subprocess
    port = args.port or 8501
    print(f"Starting Streamlit dashboard on http://localhost:{port}")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(port)])


def cmd_seed_db(args: argparse.Namespace) -> None:
    """Seed database with synthetic events and risk intelligence."""
    from src.pipeline import CyberRiskPipeline

    pipeline = CyberRiskPipeline()
    count = pipeline.seed_database(n_records=args.count)
    print(f"\nSuccessfully seeded {count} security events with risk scores and recommendations into database.")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI-Based Cyber Risk Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # generate-synthetic
    gen_parser = subparsers.add_parser("generate-synthetic", help="Generate synthetic security data")
    gen_parser.add_argument("--output", type=str, default=None, help="Output directory")
    gen_parser.set_defaults(func=cmd_generate_synthetic)

    # run-pipeline
    pipe_parser = subparsers.add_parser("run-pipeline", help="Run the full pipeline on real datasets")
    pipe_parser.add_argument("--synthetic", action="store_true", help="Use synthetic data fallback instead of real datasets")
    pipe_parser.add_argument("--sample-size", type=int, default=None, help="Sample size per raw dataset file")
    pipe_parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization")
    pipe_parser.add_argument("--no-anomaly", action="store_true", help="Skip anomaly detection")
    pipe_parser.add_argument("--no-rl", action="store_true", help="Skip RL training")
    pipe_parser.add_argument("--save-models", action="store_true", help="Save trained models")
    pipe_parser.add_argument("--save-db", action="store_true", help="Persist results into database")
    pipe_parser.set_defaults(func=cmd_run_pipeline)

    # train
    train_parser = subparsers.add_parser("train", help="Train models on real datasets")
    train_parser.add_argument("--synthetic", action="store_true", help="Use synthetic data fallback instead of real datasets")
    train_parser.add_argument("--sample-size", type=int, default=None, help="Sample size per raw dataset file")
    train_parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization")
    train_parser.add_argument("--save-models", action="store_true", help="Save trained models")
    train_parser.set_defaults(func=cmd_train)

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate saved models")
    eval_parser.add_argument("--models-dir", type=str, default="models", help="Models directory")
    eval_parser.set_defaults(func=cmd_evaluate)

    # api
    api_parser = subparsers.add_parser("api", help="Start the API server")
    api_parser.add_argument("--host", type=str, default=None, help="Host to bind")
    api_parser.add_argument("--port", type=int, default=None, help="Port to bind")
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    api_parser.set_defaults(func=cmd_api)

    # rl-train
    rl_parser = subparsers.add_parser("rl-train", help="Train the Q-Learning agent")
    rl_parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    rl_parser.add_argument("--eval-episodes", type=int, default=100, help="Number of evaluation episodes")
    rl_parser.add_argument("--save", type=str, default=None, help="Path to save the Q-table")
    rl_parser.set_defaults(func=cmd_rl_train)

    # anomaly
    anomaly_parser = subparsers.add_parser("anomaly", help="Run anomaly detection")
    anomaly_parser.set_defaults(func=cmd_anomaly)

    # risk
    risk_parser = subparsers.add_parser("risk", help="Compute risk scores")
    risk_parser.set_defaults(func=cmd_risk)

    # ethics
    ethics_parser = subparsers.add_parser("ethics", help="Run ethical AI, bias, privacy, and adversarial security audit")
    ethics_parser.set_defaults(func=cmd_ethics)

    # multi-agent
    ma_parser = subparsers.add_parser("multi-agent", help="Run multi-agent coordinated response analysis")
    ma_parser.set_defaults(func=cmd_multi_agent)

    # streamlit
    st_parser = subparsers.add_parser("streamlit", help="Launch interactive Streamlit dashboard")
    st_parser.add_argument("--port", type=int, default=8501, help="Port to bind Streamlit server")
    st_parser.set_defaults(func=cmd_streamlit)

    # seed-db
    seed_parser = subparsers.add_parser("seed-db", help="Seed database with synthetic events and risk intelligence")
    seed_parser.add_argument("--count", type=int, default=150, help="Number of events to seed")
    seed_parser.set_defaults(func=cmd_seed_db)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()