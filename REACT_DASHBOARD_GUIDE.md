# React Cyber Risk Dashboard Guide

This guide explains how to start the React dashboard, what each of its seven tabs displays, what its buttons do, and which values are live API results versus fixed demonstration values.

The dashboard is a local analyst interface for exploring cybersecurity telemetry and example detection workflows. It is not connected to a live network sensor by itself, and the remediation recommendations do not automatically block users, hosts, or IP addresses.

## Start the dashboard

Run the backend and frontend in separate terminals from the project directory.

### Terminal 1: API

```bash
cd "/Users/vivekkumar/AI-BASED CYBER RISK DETECTION SYSTEM"
source .venv311/bin/activate
python main.py api --port 8001 --reload
```

The API should report healthy at [http://localhost:8001/health](http://localhost:8001/health). Its interactive documentation is at [http://localhost:8001/docs](http://localhost:8001/docs).

### Terminal 2: React

```bash
cd "/Users/vivekkumar/AI-BASED CYBER RISK DETECTION SYSTEM/frontend"
npm install
npm run dev
```

Open [http://localhost:5173/](http://localhost:5173/). `npm install` is needed the first time, or after frontend dependencies change. Leave both terminals running while using the dashboard.

The port numbers matter: this project's API uses **8001** because another local application may use 8000. The React development server uses **5173**.

## What the dashboard does

The React application sends requests to the FastAPI backend. The backend can read event and risk-score records from the configured database, compute example risk assessments, run anomaly detection, train models and a Q-learning agent, and run an ethics audit. Some panels instead use values written directly into the frontend source.

When I inspected the shared Executive Threat Ops page, it showed **20 total events, 0 high/critical, 0 medium, and 20 low/minimal**. These are the values visible at that moment, not guaranteed permanent values.

The dashboard uses the following risk terminology:

| Risk level | Configured score interval |
|---|---:|
| Minimal | Below 0.30 |
| Low | 0.30 to below 0.60 |
| Medium | 0.60 to below 0.80 |
| High | 0.80 to below 0.90 |
| Critical | 0.90 and above |

The score is a weighted risk index in the range 0–1. The interface sometimes calls it an “exploitation probability,” but it should not be interpreted as a calibrated real-world probability.

## Sidebar and shared layout

- **CYBERSHIELD AI / SOC Autonomous Defense**: Product branding only.
- **Defense Modules**: Navigation links to the seven tabs described below. Selecting a link changes the React route; it does not launch a separate service.
- **Engine Status — ACTIVE**: A visual status badge in the layout. It is hard-coded and does not check whether the API, models, sensors, or other services are actually running.
- **CICIDS+UNSW+BETH / Multi-Modal**: Descriptive labels, not a live feed or a data-source connection indicator.

## 1. Executive Threat Ops (`/`)

**Purpose:** Present a summary of stored events and recorded risk scores.

### Header

- **LIVE SOC OPS** and **Multimodal Threat Stream**: Branding and description. The page does not subscribe to a continuous event stream.
- **Refresh Telemetry**: Calls `GET /dashboard` again and redraws this page's database summary. It does not refresh the other tabs or ingest new telemetry.

### Four summary cards

- **Total Ingested Events**: Number of recent events returned by the API (up to 50). The backend method currently used by this endpoint returns recent events for this count.
- **Critical & High Threats**: Sum of the database's `high` and `critical` risk-score counts.
- **Suspicious / Medium**: Count of `medium` risk scores.
- **Normal / Low Risk**: Sum of `low` and `minimal` risk scores.

These counts are database-backed when the API is working. The total uses up to 50 recent events, while the risk distribution is counted from recorded risk-score rows; because those queries do not necessarily cover the same set of records, category counts may not add up to the total. A value of zero may mean that no matching risk scores have been recorded; it does not prove that the environment is safe.

### Threat Severity Spectrum

Shows the `critical`, `high`, `medium`, and combined `low`/`minimal` counts as bars. Bar lengths are each category's share of the total event count.

### AI Defense Stack

Displays status-style cards for a supervised ensemble, anomaly detectors, and the RL response agent. The readiness labels and benchmark text shown here are fixed frontend copy; they are not queried from the API and should not be used as a health check.

### Live Threat Telemetry Feed

The table is intended to show recent event ID, source, origin IP, attack vector, risk level, and risk score. The backend response currently supplies fields such as `event_id`, `source_type`, `event_type`, `risk_label`, `risk_score`, and `risk_level`.

There is a frontend/backend field mismatch: the table looks for `id`, `source_ip`, `attack_type`, and a nested `risk_scores` value. When these are absent it substitutes example values (including `192.168.1.105`, `CICIDS2017`, and a risk inferred only from `risk_label`). Consequently, parts of this table can be illustrative rather than the actual event fields or risk score. It shows at most 10 rows even though its heading says “Latest 50 Captured Events.”

The page fetches its summary when opened and when **Refresh Telemetry** is clicked; it does not poll continuously despite “LIVE” wording.

## 2. Multimodal Risk Fusion (`/risk`)

**Purpose:** Change a sample event's feature values and request a fused risk score.

### Interactive Host & Network Signals

The form starts with 20 example numeric fields, including:

- Authentication: attempts, success, hour, weekend/unusual-hour flags, and failed attempts.
- Host activity: CPU and memory usage, file operations, registry changes, process/privilege flags.
- Network and access behavior: network connections, files and sensitive files accessed, data transferred, applications, device/location novelty, and login frequency.

The values are demonstration inputs held in the page state. They are not automatically populated from a selected database event. Edit the number fields to try a different scenario.

- **Reset Defaults**: Restores the original sample values in the form. It does not reset the API or database.
- **Compute Fused Risk**: Sends the edited fields to `POST /assess-risk`. The page also supplies fixed context values: source type `cicids2017`, event type `auth`, CVSS 8.8, asset criticality 90, threat score 0.85, incident count 4, and event age 0.8 hours.

### Results

- **Fused Risk Level / score**: The backend combines supervised, anomaly, and contextual scores and maps the result to a risk category. The number is shown as a percentage and a horizontal gauge.
- **Tri-Pillar Evidence Decomposition**: Displays the three input scores—supervised model, anomaly, and contextual asset impact—as percentages and bars. These are component scores, not three independent probabilities.
- **Color**: Red at scores of 0.70 and above, yellow at 0.40–0.69, and green below 0.40. These visual breakpoints are not identical to the configured risk-level thresholds.

### Important implementation detail

If no models have been trained in the current API process, the backend starts the supervised component at **0.5**. Training through the model tab populates an in-memory model registry; restarting the API clears that registry unless models are explicitly loaded elsewhere.

There is also a weight discrepancy to be aware of: the page labels the components **40% / 35% / 25%**, and the README describes those weights, but the current `config.yaml` and risk-fusion implementation use **40% / 30% / 30%**. The API's configured values determine the actual calculation.

## 3. Supervised & DNN Lab (`/models`)

**Purpose:** Show a model-comparison screen and offer a model-training action.

### Benchmark cards and table

The cards highlight a Random Forest/soft-voting result, a PyTorch neural-network result, and the claimed data-split protocol. The table lists a subset of model names with accuracy, precision, recall, F1 score, and a descriptive role.

**The benchmark numbers and table are hard-coded in the React page.** They are not the metrics returned by the running backend. The screen says “11 Active Classifiers,” but the table renders seven rows. It should be read as a sample benchmark display, not as a live result for the current machine or dataset.

### Button: Retrain All 11 Models

Calls `POST /models/train`. The current API implementation generates synthetic datasets (1,000 records per generated dataset), preprocesses them, and trains the enabled models in memory. Although the success message says “real intrusion samples,” this endpoint currently uses synthetic data. It does not update the hard-coded benchmark table with measured metrics. Training can take time and requires the API to stay running.

On opening the page, it also calls `GET /models` to retrieve the available and trained model names. Those names are stored in page state but are not currently rendered as a visible live model list.

## 4. Zero-Day Anomaly Radar (`/anomalies`)

**Purpose:** Submit one sample feature vector to the anomaly detectors.

### Suspicious Vector Signals

The page starts with 20 pre-filled numerical values, such as failed logins, unusual time, high CPU/memory, large data transfer, and new-device/location indicators. These values are a local test scenario; the form does not select or edit a database event.

Edit the numeric inputs to construct another sample.

### Button: Execute Anomaly Scan

Calls `POST /detect-anomalies` with the current feature values. The API passes the numeric vector to the configured anomaly detector collection and returns a result per detector.

### Detector result cards

For each returned detector, the page shows:

- Its detector name.
- **ANOMALY DETECTED** or **NORMAL PATTERN**, based on the returned anomaly flag.
- An **Outlier Deviation Score** percentage and a bar.

This is a single-request simulation, not a continuous zero-day monitor. The score is detector-specific and is not the same thing as the fused risk score on the Risk Fusion page.

## 5. Autonomous RL Agent (`/rl`)

**Purpose:** Inspect the Q-learning agent's API-reported status and train it in the simulated environment.

### Status cards

When the page opens, it calls `GET /rl/status`. The cards are intended to show:

- **Q-Agent Status**: Whether the API's in-memory Q-table has been populated, plus episodes completed.
- **Exploration Rate (ε)**: The current agent's exploration parameter.
- **Containment Win Rate**: A result from the most recent training/evaluation in this page, if available.
- **Q-Table State Space**: Number of state entries in the current Q-table.

Some cards use fallback display values if the API has not returned a value: ε can appear as 0.05, the Q-table size as 512, and the win rate as 96%. Those fallback numbers are not proof the agent has trained or achieved that performance.

### Training Episodes and Train RL Agent button

- **Training Episodes**: Number of episodes to request; the initial value is 1,000. The input visually suggests a range from 100 to 10,000.
- **Train RL Agent (N Episodes)**: Calls `POST /rl/train`, requesting evaluation as well. The backend creates a fresh Q-learning agent for each request, trains it in the simulated environment for the requested number of episodes, and evaluates it over 100 episodes.

### Training results

After a successful response, the page shows average and peak episode reward, mean episode length, and policy win rate. These are simulation metrics, not evidence that real machines or user accounts have been contained. Training state is held in the running API process and may be lost when that process restarts.

## 6. Automated Remediation (`/recommendations`)

**Purpose:** Demonstrate a risk assessment and show ranked response suggestions.

### Button: Synthesize Remediation Playbook

Calls `POST /assess-risk` with a fixed, high-severity example scenario (including 25 attempts, zero successful logins, high resource/file activity, CVSS 9.2, asset criticality 95, threat score 0.92, five past incidents, and an age of 0.3 hours). The page does not currently let you enter or select a different incident here.

### Event Context Card

Shows a brute-force/privilege-escalation scenario name, the returned risk score and level, and the example origin IP `192.168.1.105` / host `Core-Database-Server`. The displayed IP and host name are fixed UI text; they are not taken from the API response.

### Ranked Countermeasure Playbook

Shows recommendations returned by the backend, with action, description, and match score. Actions may include monitoring, alerting, blocking an IP, quarantining a host, resetting credentials, patching, investigating, or escalating.

**These are recommendations only.** Clicking the button does not execute the recommended action or integrate with a firewall, endpoint agent, identity provider, or ticketing system.

## 7. Ethics & Defense Audit (`/ethics`)

**Purpose:** Run the fairness, privacy, and adversarial-robustness audit pipeline.

### Button: Run Ethics & Security Audit

Calls `GET /ethics/audit`. The current backend creates a pipeline, runs data preparation using 1,500 samples per source, trains models, evaluates them, calculates risk fusion, and then runs the ethics audit. This can be substantially slower than a simple status request. It requires the backend's data, Python packages, and training routines to work.

### Results

- **Algorithmic Fairness**: Shows the audit status and demographic-parity-related metric returned by the pipeline.
- **Privacy & Anonymity**: Shows achieved k-anonymity and a differential-privacy ε value from the audit.
- **Adversarial Defense**: Shows the robustness score returned by the audit.

The cards contain fallback numbers if those response fields are missing or zero (for example k=4500, ε=1.0, robustness=98.8%). The five **Jitter ε** retention cards below them are calculated locally from a fixed 98.8 baseline; they are not per-ε measurements returned by the backend. Do not treat those particular numbers as experimental audit results.

The audit page does not have a separate reset button. Click the audit button again to submit a new audit request.

## What is live, what is illustrative?

| Dashboard item | Current behavior |
|---|---|
| Sidebar Engine Status and data-source labels | Static decoration; not a service-health check |
| Executive summary counts and risk distribution | Read from the API/database when the page loads or is refreshed |
| Executive incident feed | Database-backed event list, but several visible columns use mismatched field names and example fallbacks |
| Risk Fusion inputs | Editable sample values; request and resulting scores are API-backed |
| Model benchmark figures | Fixed frontend values, not fetched training metrics |
| Model training button | Trains the backend's generated synthetic data in memory |
| Anomaly Radar inputs | Editable sample values; per-detector response comes from the API |
| RL status/training metrics | API-backed, with hard-coded UI fallbacks in some status cards |
| Remediation event story | Fixed example context; score and recommendation list come from the API |
| Remediation action | Suggestions only; no action is actually deployed |
| Ethics fairness/privacy/robustness | Requested from the API, but some displayed fallback and jitter figures are fixed in the UI |
| “LIVE” labels | Do not mean the React app receives a continuous sensor stream |

## Terms used on the pages

- **Supervised model:** Learns from labeled examples such as benign versus suspicious events.
- **Anomaly detector:** Looks for patterns that differ from its learned or configured baseline; an anomaly is not automatically a confirmed attack.
- **Contextual score:** Adds incident context such as vulnerability severity, asset criticality, threat score, prior incidents, and event age.
- **Risk fusion:** Combines multiple score components into one prioritized index.
- **Precision:** Of alerts marked positive, the fraction that were truly positive in an evaluation.
- **Recall:** Of truly positive examples, the fraction found by the model.
- **F1 score:** A combined precision/recall measure.
- **Q-learning:** A reinforcement-learning method that learns a policy from reward feedback in a simulated environment.
- **k-anonymity:** A dataset property that checks whether records are indistinguishable from at least k records on selected quasi-identifiers.
- **Differential privacy ε:** A privacy parameter; smaller ε generally means stronger privacy but more noise. Its presence on a card alone does not establish that a particular output was released with a formally reviewed privacy guarantee.

## Relevant implementation and project files

- The route-to-tab mapping is in `frontend/src/App.tsx`.
- Sidebar branding, navigation, and the static Engine Status card are in `frontend/src/components/Layout.tsx`.
- Each tab is implemented in `frontend/src/pages/`.
- The frontend's API calls and API base URL are in `frontend/src/api/client.ts`.
- API routes and request handling are in `src/api/main.py`.
- Risk weights and thresholds are configured in `config.yaml`.
- The project architecture, datasets, model overview, and startup instructions are in `README.md`.
