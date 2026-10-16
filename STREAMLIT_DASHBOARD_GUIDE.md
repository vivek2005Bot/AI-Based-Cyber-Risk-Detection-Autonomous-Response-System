# Streamlit Cyber Risk Dashboard Guide

This guide explains how to launch and use the Streamlit dashboard at `http://localhost:8501/`. It documents the eight sidebar pages, nested tabs, interactive controls, and important differences between computed results and illustrative fallback values.

The dashboard is an interactive analysis and demonstration interface. It is not a network sensor and does not stream live SOC telemetry, block traffic, quarantine hosts, reset credentials, or apply recommendations to external systems.

## Start the dashboard

From the project root, use a terminal:

```bash
cd "/Users/vivekkumar/AI-BASED CYBER RISK DETECTION SYSTEM"
source .venv311/bin/activate
streamlit run app.py
```

Open [http://localhost:8501/](http://localhost:8501/). Keep the terminal running while using the app. Streamlit runs the Python application locally; the Streamlit pages do not require the React dashboard or FastAPI server to render, though any separately launched service might be used by other tools.

## How Streamlit interaction works

The sidebar contains a **Dataset Engine** selector and **Navigation** radio buttons. Most widget changes rerun the Python script and redraw the selected page. The principal pipeline is decorated with `st.cache_resource`: when initialized successfully, the expensive trained pipeline object is cached by its function arguments for reuse during that Streamlit process. The first initialization can take a while because it loads/generates data, preprocesses it, trains models, evaluates them, runs anomaly detection and risk fusion, trains the RL agent, and runs ethics auditing.

The default dataset mode is **Real Security Datasets (CICIDS2017 + UNSW-NB15 + BETH)**. Choosing **Synthetic Telemetry Benchmark Generator** makes the Executive page request a synthetic pipeline instead. However, several other pages call the cached pipeline without passing the selected mode and therefore use the default real-data mode. Do not assume the selector switches every page.

When I inspected the shared browser page, the Executive dashboard showed **Real Telemetry**, **6,000 events analyzed**, **1,772 threats/attacks**, **0.15 mean risk**, and **0.9719 best model F1**. These are the values visible during that inspection only; they can change if the data, selected mode, pipeline results, or fallback conditions differ.

## Sidebar and navigation

- **CyberRisk AI / Multimodal Cyber Risk Detection & Response**: Dashboard branding.
- **Dataset Engine**:
  - **Real Security Datasets (CICIDS2017 + UNSW-NB15 + BETH)**: Requests real-data integration for the Executive dashboard pipeline. The pipeline uses a sample limit of 2,000 per source.
  - **Synthetic Telemetry Benchmark Generator**: Requests generated synthetic data for the Executive dashboard pipeline (2,000 records per generated dataset).
  - This changes a pipeline function argument and causes a Streamlit rerun; it does not download new files or switch every page. The sidebar's green **Active Mode** message only reflects the selected label, not successful data loading.
- **Navigation**: Select one of the eight pages explained below. This is the main navigation for the Streamlit app.
- **Framework information**: A static informational label, not a runtime version or health report.

## 1. Executive Threat Dashboard

**Purpose:** Show a high-level summary of the cached pipeline's dataset, model evaluation, risk fusion, and examples from the processed data.

### KPI row

- **Total Events Analyzed**: Number of rows in the pipeline's processed dataset.
- **Threats / Attacks**: Sum of the `is_attack` labels in that dataset.
- **Mean Risk Score**: Mean fused risk from the pipeline result.
- **Best Model F1-Score**: Maximum F1 score found in the model evaluation results.
- **RL Agent Status**: Currently rendered as **Trained & Active / Optimized** unconditionally; it is not a live check of the RL agent.

The first four figures are based on the pipeline when those results are present. The code supplies demo fallback numbers if relevant data is unavailable (for example, fallback event/attack totals, mean risk, or F1). Check the underlying pipeline or logs if numbers look unexpected.

### Risk Severity Distribution

A pie chart of the pipeline's fused-risk category counts. If the pipeline result is missing, the page uses a fixed example distribution; if the computed distribution totals zero, it substitutes another fixed example distribution. A fallback pie is not a measurement of current traffic.

### Multimodal Signal Contributions

A bar chart of the configured fusion weights for the supervised prediction, anomaly score, and contextual indicators. By default the project's current configuration is 0.40 / 0.30 / 0.30. This chart shows weights, not the amount each signal contributed to an individual event.

### Recent Security Events Stream

Despite the word “Stream,” this is not a live event stream. The page displays up to the first 15 rows from the pipeline's in-memory processed dataset, selecting available columns such as source type, event type, attack type, and attack label. It does not currently display the `events` list fetched from the database; that query is unused in this page.

## 2. Data & Preprocessing Hub

This page has four nested tabs.

### Source Datasets Preview

- **Select Telemetry Domain**: Chooses a preview dataset: CICIDS2017, UNSW-NB15, BETH authentication, or one of three generated synthetic sets when those datasets loaded successfully.
- **Total Records Preview / Columns**: Dimensions of the selected preview DataFrame.
- **Table**: First 10 rows of that preview dataset; this is a sample, not the entire dataset.

The preview loader reads up to 500 samples per real source file and caches the loaded results. Synthetic previews are generated separately. The loader catches failures silently; a missing/unreadable dataset may be omitted without explaining the cause, and an empty list of preview datasets can prevent the selector from rendering.

### Class Imbalance (SMOTE)

Shows the original `is_attack` label counts and a purported post-SMOTE comparison. The “after” chart is constructed by setting both class counts equal to the original majority count; it is a visualization of an idealized balanced distribution, not the actual resampler output or measured number of generated samples. The core model pipeline applies imbalance handling to the training split only, but this chart does not inspect that returned resampled training set.

### Feature Selection

Shows up to 15 feature names from the cached pipeline (or example feature names if unavailable). The displayed “Information Gain” values are generated as a descending linear sequence between 0.85 and 0.25, not extracted from each feature's actual mutual-information or ANOVA score. Treat this chart as illustrative ranking artwork, not a measured importance report.

### Dimensionality Reduction (PCA/LDA/SVD)

If a test set exists, the page takes up to its first 300 rows, fits a fresh 3-component PCA projection, and renders a 3D scatter plot colored by the corresponding attack/normal labels. The displayed explained-variance total comes from that PCA fit. Although the tab title mentions PCA/LDA/SVD, this visualization currently runs PCA only.

## 3. Supervised & Deep Learning Lab

**Purpose:** Inspect the evaluation results created by the cached pipeline and try a small manual inference scenario.

### Model metrics table and F1 chart

When the pipeline has evaluation results, the page displays each model's available accuracy, precision, recall, F1, ROC-AUC, and Matthews correlation coefficient, highlighting the maximum in each metric. The bar chart compares model F1 scores. These values come from the pipeline evaluation on its held-out test data, not from an external benchmark service.

The page heading says “8 supervised models,” while the actual displayed rows depend on enabled/trained models and ensembles in the pipeline.

### Real-Time Event Risk Predictor controls

Four sliders set:

- **Failed Attempts (24h)**: 0–100, default 12.
- **CPU Usage (%)**: 0–100, default 78.5.
- **Memory Usage (%)**: 0–100, default 84.
- **Active Network Connections**: 1–200, default 85.

### Button: Run AI Risk Inference

Constructs an input vector of zeros and places those four slider values in its first four positions, then asks the first model in the pipeline's trained-model collection for a prediction probability if it supports `predict_proba`. It shows a suspicious probability and labels scores above 60% suspicious.

This is a rough demonstration, not a full event-ingestion flow: the input vector is not assembled by feature name, and the default zeros may not represent a realistic event or the original training feature scales. If there is no suitable trained model, a simple threshold rule (`attempts > 5`) supplies 82% or 15%. The “Real-Time” title does not mean it is processing a live event stream.

## 4. Unsupervised Anomaly & Clustering

**Purpose:** Show clustering/anomaly analysis results and a 2D projection of behavioral clusters.

### Clustering Quality Metrics

Displays the pipeline's clustering metrics if present. Otherwise, it shows fixed example K-Means Silhouette and Calinski-Harabasz figures.

### Outlier Detection Rates

Displays the pipeline's anomaly results if present. Otherwise, it shows fixed example 10% Isolation Forest and LOF rates.

### Clustering Structure Visualization

If training data exists, the page fits a four-cluster K-Means model to its first 400 rows, fits PCA to reduce those rows to two dimensions, and plots the points by cluster. This visualization is calculated when the page runs; it is not a live cluster monitor. K-Means cluster IDs are arbitrary group labels, not severity levels.

## 5. Evidence-Based Risk Fusion

**Purpose:** Interactively calculate a weighted score from three user-selected component scores. This page performs the arithmetic locally; it does not send an event to the API or automatically use an actual prediction.

### Weight sliders

- **Supervised Model Weight (w1)**: default 0.40.
- **Anomaly Detector Weight (w2)**: default 0.30.
- **Contextual Indicators Weight (w3)**: default 0.30.

The app normalizes these values by their sum and displays the normalized weights. If all three are set to zero, it uses 0.40 / 0.30 / 0.30 as the calculation fallback.

### Simulate Incoming Security Signal sliders

- **Supervised ML Score**: default 0.75.
- **Unsupervised Anomaly Score**: default 0.65.
- **Contextual Score (CVSS / Criticality)**: default 0.80.

Each score ranges from 0 to 1. The fused score is:

`normalized supervised weight × supervised score + normalized anomaly weight × anomaly score + normalized context weight × context score`

With the initial values, the score is **0.735**, categorized as **HIGH** by this page's own tiers.

### Results

- **Fused Cyber Risk Score**: Weighted score from the sliders.
- **Severity Tier**: This page uses its own thresholds: Minimal below 0.30, Low 0.30–0.49, Medium 0.50–0.69, High 0.70–0.84, Critical at or above 0.85. These differ from the FastAPI risk-fusion configuration.
- **Fused Cyber Risk Index gauge**: Visual display of the same score from 0 to 1.
- **Signal Breakdown & Attribution pie**: Shows each weighted component's share of the calculated score; it does not explain individual feature-level causes.

There is no submit button: Streamlit recalculates the formula and redraws the page when sliders change.

## 6. Remediation & Recommendation

**Purpose:** Show response suggestions for a selected example event type and risk level.

- **Event Domain**: Select `network`, `auth`, `endpoint`, or `behavior`.
- **Risk Level Category**: Select `critical`, `high`, `medium`, `low`, or `minimal`.
- **Suggested Mitigation Actions**: The hybrid recommender generates and displays up to four ranked actions and descriptions for the selected values. Changing either selection reruns the app and updates the recommendations.

There is no separate **Recommend** button. The recommender is instantiated within the Streamlit page. These are suggestions only: nothing is automatically blocked, patched, quarantined, or otherwise deployed, and the page does not persist the actions as incident records.

## 7. RL & Multi-Agent Operations

This page contains two nested tabs.

### Single-Agent Q-Learning

- **Win / Convergence Rate**, **Mean Cumulative Reward**, **Avg Episode Steps**: Read from the pipeline RL evaluation when available. If a field is missing, the page shows demo defaults (92%, 14.5, and 1.8).
- **Simulated Q-Learning Convergence Curve**: A 100-point curve created directly in the page using a smooth formula plus random noise. It is not plotted from the pipeline's actual training episode history, so it can vary on reruns and should not be interpreted as measured convergence.
- **Training control**: There is no training button or episode-count input on this page. The RL agent was trained for 200 episodes as part of the cached pipeline initialization.

### Multi-Agent Coordination System

Describes the Network Analyst, Identity Analyst, Endpoint Analyst, and Decision Agent roles. The app constructs a multi-agent system and analyzes a fixed six-value example state each time the tab renders. It shows three specialist signal values and the decision agent's recommended action. These are simulated outputs for the fixed state, not signals collected from real network, identity, or endpoint agents.

## 8. Ethical AI, Privacy & Security

This page has three nested tabs. It reads the ethics results from the default cached pipeline; there is no button on this page to rerun only the audit.

### Algorithmic Fairness

- **Fairness Compliance Status**: Displays the pipeline audit's status when available, with a `COMPLIANT` fallback if missing.
- **Group metrics table**: Appears when audit group metrics exist.
- **Four-Fifths Rule Check**: The explanatory information message always says ratios conform to the 0.80–1.25 corridor. It is static text and is not conditional on the measured audit result; inspect the actual status/table instead.

### Privacy & Anonymization

- **K Achieved**: k-anonymity result, or a fallback of 5 if absent.
- **Re-identification Risk Rate**: Audit rate formatted as a percentage, with a zero fallback.
- **DP Mean Risk**: Differentially private mean-risk result, or a fallback of 0.38 if absent.
- **Privacy Budget (Epsilon)**: Epsilon from the audit, or a fallback of 1.0.

Fallback values are presentation defaults and do not mean that the audit ran successfully or that a formal privacy guarantee was independently verified.

### Adversarial Robustness

- **Model Robustness Score**: Audit result, or a fallback of 84.5%, with a static “Resilient” delta label.
- **Perturbation table and line chart**: Shown only when the audit returns perturbation-analysis results. They plot evasion success by perturbation epsilon.

## What is computed, cached, or illustrative?

| Item | Current behavior |
|---|---|
| Main analysis pipeline | Expensive data/model/analysis pipeline, cached as a Streamlit resource after successful initialization |
| Dataset Engine selector | Selects the Executive dashboard's pipeline mode; does not consistently change all pages |
| Executive KPIs | Derived from pipeline data/results when available, with fixed fallback values |
| Executive risk pie | Uses pipeline risk distribution, with fixed example distributions as fallbacks |
| Executive event table | First 15 processed pipeline rows, not a live event feed; a separate database event query is unused |
| Dataset previews | Sample rows from real or generated data, separately cached; loading failures can be silently omitted |
| SMOTE “after” chart | Idealized equal counts, not actual post-resampling counts |
| Feature “Information Gain” | Descending placeholder values, not calculated feature importance |
| PCA projection | Computed from the current pipeline's first test rows; PCA is used despite the broader tab title |
| Model metrics table | Pipeline evaluation results |
| Manual inference | Four slider values inserted into the first four positions of a zero-filled vector; simple fallback if no suitable model |
| Anomaly metrics | Pipeline results where present; otherwise fixed example values |
| Risk Fusion Simulator | Local arithmetic using editable sliders; not an API or database query |
| Remediation suggestions | Recommender output for selected example categories; no response action deployed |
| RL convergence chart | Synthetic curve with random noise, not episode-history data |
| Multi-agent signals | Simulated analysis of a fixed example state |
| Fairness compliance message | Static text; it can disagree with actual audit status |
| Ethics fallback values | Display defaults when result fields are absent; not evidence of successful measurement |

## Terms used on the pages

- **CICIDS2017 / UNSW-NB15 / BETH**: Public or research cybersecurity datasets used for network intrusion, traffic, or host/authentication analysis, subject to what is actually available in the local data directories.
- **SMOTE**: A training-data oversampling method that synthesizes minority-class examples; it does not create new real attack observations.
- **PCA**: Principal Component Analysis, a method for projecting many numeric features into fewer dimensions for visualization or modeling.
- **Precision**: Among examples classified as positive, the fraction that were positive.
- **Recall**: Among true positive examples, the fraction detected.
- **F1 score**: A combined precision/recall metric.
- **Anomaly**: A pattern that differs from a detector's baseline; it is not automatically a confirmed attack.
- **Risk fusion**: A weighted combination of model, anomaly, and context scores.
- **Q-learning**: A reinforcement-learning algorithm trained from reward feedback in a simulated environment.
- **k-anonymity**: A property of a dataset under selected quasi-identifiers; each record's group should contain at least k indistinguishable records.
- **Differential privacy ε**: A privacy parameter; smaller epsilon generally provides stronger privacy at the cost of greater noise. Displaying epsilon alone does not establish that a release has a formally reviewed guarantee.

## Related files

- Streamlit page and widget implementation: `app.py`.
- Data integration, training, and analysis logic: `src/pipeline.py`.
- Feature preprocessing configuration: `config.yaml`.
- Overall architecture, datasets, and project setup: `README.md`.
