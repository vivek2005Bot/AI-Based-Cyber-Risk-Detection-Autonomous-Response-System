# 🛡️ AI-Based Cyber Risk Detection & Autonomous Response System

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.2-61dafb.svg)](https://react.dev/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.3-38bdf8.svg)](https://tailwindcss.com/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-RL-black.svg)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **An enterprise-grade, end-to-end cybersecurity intelligence platform combining multimodal log acquisition, leak-free machine learning, deep neural networks, unsupervised zero-day anomaly detection, contextual risk fusion, reinforcement learning for automated containment, and algorithmic ethics/adversarial defense audits.**

---

## 📑 Table of Contents
- [Executive Overview](#-executive-overview)
- [Key Architectural Highlights](#-key-architectural-highlights)
- [System Architecture](#-system-architecture)
- [Real Cybersecurity Datasets](#-real-cybersecurity-datasets)
- [Machine Learning & Deep Learning Engine](#-machine-learning--deep-learning-engine)
- [Evidence-Based Multimodal Risk Fusion](#-evidence-based-multimodal-risk-fusion)
- [Autonomous Response & Multi-Agent RL](#-autonomous-response--multi-agent-rl)
- [Ethical AI, Privacy & Adversarial Defense](#-ethical-ai-privacy--adversarial-defense)
- [User Interfaces (Streamlit & React)](#-user-interfaces-streamlit--react)
- [Installation & Quickstart](#-installation--quickstart)
- [Project Directory Structure](#-project-directory-structure)
- [CLI Reference](#-cli-reference)
- [API Endpoints](#-api-endpoints)
- [Verification & Testing](#-verification--testing)
- [License & Authors](#-license--authors)

---

## 🎯 Executive Overview

Modern enterprise Security Operations Centers (SOCs) face an overwhelming volume of alerts across disjointed telemetry streams—network traffic, authentication records, host process operations, and threat intelligence. Manual triage leads to alert fatigue, missed zero-day attacks, and delayed containment.

This system provides a unified, intelligent framework that:
1. **Ingests & Normalizes Real Data**: Harmonizes heterogeneous streams (CICIDS2017 network flows, UNSW-NB15 modern attack vectors, BETH OS authentication kernel logs, and NVD/CVE vulnerabilities).
2. **Predicts Threats with 11 ML/DL Models**: Uses leak-free preprocessed training across classical algorithms, deep neural nets, and weighted soft voting ensembles.
3. **Detects Novel / Zero-Day Outliers**: Runs unsupervised Isolation Forests and Local Outlier Factor (LOF) to flag anomalies without requiring signatures.
4. **Calculates Contextual Risk**: Fuses supervised confidence (40%), anomaly scores (35%), and CVSS asset criticality (25%) into an actionable probability index ($0.0 - 1.0$).
5. **Autonomously Prescribes Countermeasures**: Recommends mitigation playbooks using hybrid filtering and trains a Gymnasium Q-Learning RL agent for automated containment.
6. **Audits Ethical AI & Security**: Quantifies demographic parity across user cohorts, enforces $k$-anonymity & Differential Privacy ($\varepsilon$-budget), and tests models against adversarial evasion attacks.

---

## 🚀 Key Architectural Highlights

* **100% Leak-Free ML Pipeline**: Splitting occurs *prior* to SMOTE oversampling and standard scaling, preventing artificial metric inflation and ensuring generalizable real-world performance (~86% accuracy, ~0.89 precision).
* **Deep Neural Network Integration**: Built with PyTorch using dropout regularization, weight decay, and automatic hardware acceleration (`MPS` on Apple Silicon, `CUDA` on Nvidia GPUs, `CPU` fallback).
* **Multi-Agent Specialist Architecture**: Coordinated specialist agents (Network Analyst, Endpoint Guard, Threat Intel Specialist) report to a Decision Facilitator for incident consensus.
* **Dual Modern Frontend Stacks**:
  * **Interactive SOC Analyst Suite (Streamlit)**: Complete diagnostic platform with real-time sliders, SHAP/LIME explainability, and dataset switching.
  * **Executive Cyber Matrix Dashboard (React 18 + Vite + TailwindCSS)**: High-contrast, neon Red-Yellow-Green theme with REST API live telemetry polling.

---

## 🏗️ System Architecture

```
                                 [ Raw Multi-Source Telemetry ]
                                                │
         ┌──────────────────────────────┬───────┴──────────────────────┬──────────────────────────────┐
         ▼                              ▼                              ▼                              ▼
  CICIDS2017 (Network)          UNSW-NB15 (Intrusions)        BETH (Kernel Auth)             NVD / CVE Feeds
         │                              │                              │                              │
         └──────────────────────────────┴───────┬──────────────────────┴──────────────────────────────┘
                                                ▼
                                   [ Data Integration & ETL ]
                                  (Missing Values, Scaler, PCA)
                                                │
                                    ┌───────────┴───────────┐
                                    ▼                       ▼
                              Training Set               Test Set (Held-out)
                             (SMOTE Resampling)             │
                                    │                       │
         ┌──────────────────────────┴──────────┐            │
         ▼                                     ▼            │
 [ Supervised & Deep Learning ]     [ Unsupervised Anomaly ]│
 • Logistic Regression, Naive Bayes • Isolation Forest      │
 • KNN, SVM, Decision Tree          • Local Outlier Factor  │
 • Random Forest, Extra Trees       • K-Means / DBSCAN      │
 • PyTorch 3-Layer DNN (MLP)                │               │
 • Soft Voting & Stacking Ensembles         │               │
         │                                  │               │
         └──────────────────┬───────────────┘               │
                            ▼                               ▼
                 [ Multimodal Evidence Fusion ] ◄───────────┘
               (Supervised + Anomaly + CVSS Context)
                            │
                            ├──────────────────────────────────────────────┐
                            ▼                                              ▼
            [ Autonomous Decision & Mitigation ]           [ Ethical AI & Defense Audits ]
            • Hybrid Recommendation Engine (Content+CF)    • Demographic Parity / Fairness
            • Q-Learning Policy Agent (Gymnasium)          • k-Anonymity & Differential Privacy
            • Multi-Agent Consensus Architecture           • Adversarial Evasion Stress Testing
                            │                                              │
                            └──────────────────────┬───────────────────────┘
                                                   ▼
                                       [ Dual Interface Layer ]
                               ┌───────────────────┴───────────────────┐
                               ▼                                       ▼
                     [ FastAPI REST Backend ]               [ Streamlit SOC Lab ]
                     [ React 18 Cyber Matrix ]
```

---

## 📊 Real Cybersecurity Datasets

The platform natively ingests and maps real-world security logs into a standardized schema:

| Dataset | Modality | Attack Categories Covered | Raw Location |
| :--- | :--- | :--- | :--- |
| **CICIDS2017** | Network Packet Flows | DDoS, PortScan, Botnet, Web Attacks, Infiltration, Brute Force | `data/raw/cicids2017/` |
| **UNSW-NB15** | Network Intrusion Logs | Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance | `data/raw/unsw_nb15/` |
| **BETH Authentication** | Linux Kernel Sensor Telemetry | Process injection, root escalation, credential harvesting (`evil`/`sus` flags) | `data/raw/authentication/` |
| **NVD / CVE** | Threat Intelligence | CVSS v2/v3 vulnerability scores, Common Weakness Enumeration (CWE) | `data/raw/nvd_cve/` |

---

## 🧠 Machine Learning & Deep Learning Engine

All models are trained strictly using leak-free practices:

### Model Benchmark Summary (Evaluated on held-out test split):

| Algorithm | Model Family | Accuracy | Precision | Recall | F1-Score | Key Strength |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Forest** | Tree Ensemble | **86.47%** | **89.87%** | **86.47%** | **0.8656** | High non-linear feature split |
| **Soft Voting Ensemble** | Meta-Classifier | **86.47%** | **89.87%** | **86.47%** | **0.8656** | Variance reduction & consensus |
| **PyTorch Deep Neural Net** | Deep Learning (MLP) | **86.39%** | **89.85%** | **86.39%** | **0.8652** | High generalizability with Dropout |
| **K-Nearest Neighbors** | Instance-Based | 86.45% | 89.90% | 86.45% | 0.8658 | Strong local density separation |
| **Decision Tree** | Single Estimator | 86.41% | 89.86% | 86.41% | 0.8654 | Transparent rule interpretability |
| **Naive Bayes** | Probabilistic | 86.12% | 89.41% | 86.12% | 0.8626 | Ultra-fast inference latency |
| **Support Vector Machine (RBF)**| Kernel Method | 86.14% | 89.64% | 86.14% | 0.8627 | Robust margin optimization |
| **Logistic Regression** | Linear Model | 85.16% | 89.18% | 85.16% | 0.8529 | Reliable linear baseline |

---

## ⚖️ Evidence-Based Multimodal Risk Fusion

Rather than relying purely on binary classification, the platform computes a continuous **Composite Cyber Risk Score** ($R \in [0.0, 1.0]$):

$$R = w_s \cdot P_{\text{supervised}} + w_a \cdot S_{\text{anomaly}} + w_c \cdot C_{\text{contextual}}$$

* **$P_{\text{supervised}}$ ($w_s = 0.40$)**: Probability output from the trained ML/DL ensemble.
* **$S_{\text{anomaly}}$ ($w_a = 0.35$)**: Normalized zero-day deviation score from Isolation Forest & LOF.
* **$C_{\text{contextual}}$ ($w_c = 0.25$)**: Calculated impact based on CVSS score, asset criticality, and historical incident frequency.

### Risk Level Tiers:
* 🟢 **Minimal** ($0.0 \le R < 0.2$) | 🟡 **Low** ($0.2 \le R < 0.4$) | 🟠 **Medium** ($0.4 \le R < 0.6$)
* 🔴 **High** ($0.6 \le R < 0.8$) | 🟣 **Critical** ($0.8 \le R \le 1.0$)

---

## 🤖 Autonomous Response & Multi-Agent RL

1. **Hybrid Remediation Recommender**:
   * Combines **Content-Based Filtering** (matching attack vectors to security playbooks) and **Collaborative Filtering** (historical mitigation success matrices).
   * Generates ranked actions: `Block IP`, `Quarantine Host`, `Reset Credentials`, `Patch CVE`, `Escalate to Tier 3`.
2. **Reinforcement Learning Agent (`Gymnasium` + Q-Learning)**:
   * Custom cyber war-game environment where the state represents $(R, \text{threat\_type}, \text{containment\_status})$.
   * Policy optimizes rewards: $+10$ for swift containment, $-50$ for successful data exfiltration, $-5$ for service disruption.
3. **Multi-Agent Specialist Hierarchy**:
   * **Network Specialist**: Inspects packet volumes, unusual ports, and payload signatures.
   * **Endpoint Specialist**: Analyzes process execution parentage, CPU spikes, and privilege escalations.
   * **Threat Intel Specialist**: Cross-references external CVEs and malicious IP blacklists.
   * **Decision Facilitator**: Synthesizes specialist opinions into coordinated defensive actions.

---

## 🛡️ Ethical AI, Privacy & Adversarial Defense

* **Fairness & Bias Auditing**: Evaluates Demographic Parity Ratio (DPR) and Disparate Impact across geographic or departmental cohorts to ensure legitimate business users are not disproportionately blocked ($DPR \ge 0.80$).
* **Data Privacy Protection**:
  * **$k$-Anonymity**: Evaluates quasi-identifiers (`source_ip`, `hour`, `auth_method`) ensuring $k \ge 5$.
  * **Differential Privacy ($\varepsilon$-budget)**: Laplace perturbation mechanism on published risk reports to prevent membership inference attacks.
* **Adversarial Robustness Testing**: Simulates evasion attacks (boundary jitter, feature manipulation with budgets $\varepsilon \in [0.01, 0.30]$) to confirm model stability under adversarial evasion.

---

## 💻 User Interfaces (Streamlit & React)

### 1. React 18 + Vite Cyber Matrix Dashboard
* High-contrast glassmorphism aesthetic with **Red-Yellow-Green** color grading.
* Live REST API integration with FastAPI (`/dashboard`, `/assess-risk`, `/ethics/audit`, `/rl/status`).
* Pages: *Executive Threat Ops*, *Multimodal Risk Fusion*, *Supervised & DNN Lab*, *Zero-Day Anomaly Radar*, *Autonomous RL Agent*, *Remediation Playbooks*, *Ethics & Defense Audit*.
* See [REACT_DASHBOARD_GUIDE.md](./REACT_DASHBOARD_GUIDE.md) for a tab-by-tab explanation of dashboard metrics, controls, and which displays are live versus illustrative.

### 2. Streamlit Interactive SOC Lab
* Interactive sidebar controls for full parameter experimentation.
* In-depth feature importance visualizer, SHAP/LIME explainability plots, and database event inspector.
* See [STREAMLIT_DASHBOARD_GUIDE.md](./STREAMLIT_DASHBOARD_GUIDE.md) for a page-by-page explanation of the Streamlit dashboard, its controls, and computed versus illustrative values.

---

## ⚙️ Installation & Quickstart

### Prerequisites
* **Python 3.10+** (Python 3.11 / 3.12 / 3.14 compatible)
* **Node.js 18+** & **npm** (for the React frontend)
* **Git**

### 1. Clone & Set Up Python Virtual Environment
```bash
git clone https://github.com/yourusername/ai-cyber-risk-detection.git
cd ai-cyber-risk-detection

# Create and activate virtual environment
python3 -m venv .venv
source .venv311/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline & Train Models
```bash
# Train on real datasets with leak-free evaluation
python main.py train --sample-size 2500

# Or execute the complete end-to-end pipeline
python main.py run-pipeline --sample-size 2500 --save-models --save-db
```

### 3. Launch the Backend & Frontends

#### Option A: Launch Streamlit Dashboard
```bash
streamlit run app.py
```
👉 Access Streamlit at: **[http://localhost:8501](http://localhost:8501)**

#### Option B: Launch FastAPI Backend + React Dashboard
**Terminal 1 (FastAPI Backend):**
```bash
python main.py api --port 8001 --reload
```
👉 Interactive Swagger Docs: **[http://localhost:8001/docs](http://localhost:8001/docs)**

**Terminal 2 (React Frontend):**
```bash
cd frontend
npm install
npm run dev
```
👉 Access React Dashboard at: **[http://localhost:5173](http://localhost:5173)**

---

## 📁 Project Directory Structure

```text
├── config.yaml                     # Global configuration (paths, hyperparameters, feature weights)
├── main.py                         # Unified CLI entry point for training, evaluation, and APIs
├── app.py                          # Streamlit interactive SOC analyst dashboard
├── requirements.txt                # Python dependencies
├── data/
│   ├── cyber_risk.db               # SQLite database storing processed events & risk scores
│   └── raw/                        # Raw cybersecurity datasets
│       ├── cicids2017/             # CICIDS2017 CSV files
│       ├── unsw_nb15/              # UNSW-NB15 CSV files
│       ├── authentication/         # BETH authentication logs
│       └── nvd_cve/                # NVD vulnerability intelligence
├── frontend/                       # React 18 + Vite + TailwindCSS frontend
│   ├── src/
│   │   ├── api/client.ts           # Axios REST API client
│   │   ├── components/Layout.tsx   # Sidebar & Cyber Matrix layout wrapper
│   │   └── pages/                  # Dashboard, RiskAnalysis, ModelComparison, etc.
│   ├── package.json
│   └── tailwind.config.js
├── src/                            # Core Python source modules
│   ├── anomaly/                    # Isolation Forest, LOF, K-Means, DBSCAN detectors
│   ├── api/                        # FastAPI application and route controllers
│   ├── data/                       # Data loaders, ingestion & multi-source integration
│   ├── database/                   # SQLAlchemy ORM models & repository layer
│   ├── ethics/                     # Fairness auditor, Differential Privacy, Adversarial tests
│   ├── models/                     # 11 Supervised algorithms, PyTorch DNN, Ensembles, Evaluation
│   ├── preprocessing/              # Leak-free cleaning, imputation, SMOTE, PCA, Feature Selection
│   ├── recommendation/             # Content-based, Collaborative & Hybrid remediation engines
│   ├── risk_fusion/                # Tri-pillar composite risk score fusion engine
│   ├── rl/                         # Gymnasium environment, Q-Learning agent, Multi-Agent system
│   └── utils/                      # Config loaders and logging utilities
├── tests/                          # Complete automated test suite (Pytest)
└── notebooks/                      # Step-by-step Jupyter research notebooks
```

---

## ⌨️ CLI Reference

The platform includes a CLI via [`main.py`](file:///Users/vivekkumar/AI-BASED%20CYBER%20RISK%20DETECTION%20SYSTEM/main.py):

```bash
# Train all models on real datasets
python main.py train [--sample-size N] [--optimize] [--save-models]

# Run the complete end-to-end pipeline
python main.py run-pipeline [--sample-size N] [--save-models] [--save-db]

# Train the Reinforcement Learning agent
python main.py rl-train --episodes 1000 --save models/q_table.npy

# Run ethical AI & adversarial security audit
python main.py ethics

# Run coordinated multi-agent simulation
python main.py multi-agent

# Seed database with sample telemetry events
python main.py seed-db --count 150

# Launch FastAPI backend server
python main.py api --host 0.0.0.0 --port 8001 --reload
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | System health check and model loading status |
| `GET` | `/dashboard` | Executive summary metrics and latest telemetry feed |
| `POST` | `/assess-risk` | Compute fused risk score, decomposition, and playbooks |
| `POST` | `/detect-anomalies` | Execute unsupervised Isolation Forest and LOF checks |
| `POST` | `/recommend` | Generate ranked incident remediation actions |
| `GET` | `/models` | Retrieve trained model registry and benchmark metrics |
| `POST` | `/train` | Trigger retraining of all supervised and deep models |
| `GET` | `/rl/status` | Fetch Q-Learning agent training state and win rates |
| `POST` | `/rl/train` | Train the RL containment policy agent |
| `POST` | `/rl/multi-agent` | Run multi-agent specialist consensus analysis |
| `GET` | `/ethics/audit` | Run full fairness, privacy ($k$-anonymity), and adversarial audit |

---

## 🧪 Verification & Testing

The project maintains comprehensive unit and integration test coverage across all modules:

```bash
pytest -v tests/
```

**Results:**
```text
======================== 47 passed in 9.37s ========================
- test_anomaly.py ......................... [PASSED]
- test_api.py ............................. [PASSED]
- test_config.py .......................... [PASSED]
- test_ethics.py .......................... [PASSED]
- test_pipeline.py ........................ [PASSED]
- test_preprocessing.py .................. [PASSED]
- test_pytorch_model.py ................... [PASSED]
- test_recommendation.py .................. [PASSED]
- test_risk_fusion.py ..................... [PASSED]
- test_rl.py .............................. [PASSED]
```

---

## 📄 License & Authors

This project is distributed under the **MIT License**. See `LICENSE` for details.

* **Author**: Vivek Kumar
* **Project**: AI-Based Cyber Risk Detection & Autonomous Response System
* **Architecture**: Python 3 + PyTorch + Scikit-Learn + Gymnasium + FastAPI + React 18 + TailwindCSS
