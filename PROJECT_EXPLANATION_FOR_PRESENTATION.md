# AI-Based Cyber Risk Detection System — Simple Project Explanation

This guide explains the project in everyday language so it can be presented to a teacher or someone without an IT background.

## 1. What is the main purpose of the project?

The project is a cybersecurity assistant. It studies security-event information, looks for signs of attacks, estimates how serious a risk may be, and suggests what a security team could do next.

In simple terms, it follows these steps:

1. **Collect and prepare information** about network activity, logins, computers, and known software vulnerabilities.
2. **Learn from examples** of normal and suspicious activity.
3. **Look for unusual behavior** that does not match familiar patterns.
4. **Combine the clues** into one easier-to-understand risk score.
5. **Suggest possible responses**, such as investigating, alerting, or blocking a suspicious connection.
6. **Check the system** for fairness, privacy, and resistance to attempts to fool its models.

The Streamlit and React dashboards are ways to view and demonstrate these functions. They are not, by themselves, live network-monitoring tools. In particular, the response suggestions are not automatically carried out on a real firewall or computer.

### A short explanation to say aloud

> “Our project is a cybersecurity assistant. It uses security data to learn what normal and suspicious activity can look like, flags unusual behavior, combines the evidence into a risk level, and suggests possible actions. We also test how well the system works and whether it handles data responsibly. The dashboards help us see and demonstrate the results.”

## 2. What are the six notebooks for?

Think of the notebooks as six lessons or experiments that explain different parts of the project.

| Notebook | In everyday language |
|---|---|
| **01 — Data acquisition and preprocessing** | Brings security information together, cleans it, makes it easier for models to use, handles the fact that attacks are less common than normal activity, selects useful clues, and demonstrates reducing many columns into a smaller set for analysis. |
| **02 — Supervised and deep-learning models** | Teaches several models using examples already marked as normal or attack. It compares their results and includes a neural-network model. |
| **03 — Unsupervised clustering and anomaly detection** | Looks for groups and unusual records without needing every record to have an attack label. This is useful for finding activity that looks different from the usual patterns. |
| **04 — Evidence-based risk fusion and recommendations** | Shows how model results, unusual-behavior scores, and incident context can be combined into a single risk score, then demonstrates suggested response actions. |
| **05 — Reinforcement learning and multi-agent system** | Trains a decision-making agent in a simulated cybersecurity environment using rewards and penalties. It also demonstrates specialist agents that examine different kinds of clues before a central agent chooses a response. |
| **06 — Ethical AI, privacy, and security analysis** | Examines whether model outcomes differ across groups, explores privacy methods, and tests whether small changes to input data can fool the model. |

The notebooks are experiments and demonstrations; they do not mean the system is continuously observing a live company network. Some notebook and dashboard workflows use generated example data, so check the dataset and results for the specific demonstration before describing them as real-world measurements.

## 3. Where are the learning methods used?

### Supervised learning — learning from answers

The model is shown examples that already have an answer, such as **normal** or **attack**. It learns to make a prediction for a new example.

- **Main place:** Notebook **02**, “Supervised & Deep Learning Model Benchmarking.”
- **Also used in:** The main data/model pipeline and the model-comparison or risk-analysis screens.
- **Examples in the project:** Logistic regression, decision trees and tree ensembles, support vector machines, nearest neighbors, and a PyTorch neural network.
- **Why use it?** To estimate whether a new event resembles previously labelled normal or attack examples.

### Unsupervised learning — finding patterns without answers

The method is not given a normal/attack answer for each example. Instead, it groups similar activity or flags activity that is far from common patterns.

- **Main place:** Notebook **03**, “Unsupervised Clustering & Anomaly Detection.”
- **Methods shown:** K-Means, DBSCAN/hierarchical clustering, Isolation Forest, and Local Outlier Factor (LOF).
- **Also used in:** The Streamlit clustering/anomaly page and React anomaly screen.
- **Why use it?** It can point out unusual activity, including patterns that were not specifically labelled in the training examples. An unusual event is a warning to investigate, not proof of an attack.

### Reinforcement learning — learning by trying actions and receiving rewards

An agent tries possible actions in a **simulation**. Helpful choices receive better rewards, and harmful or costly choices receive penalties. Over many attempts, it learns a policy—a guide for which action to choose in a simulated situation.

- **Main place:** Notebook **05**, using Q-learning.
- **Also used in:** The Streamlit RL page and React RL training page.
- **Why use it?** To demonstrate how a response policy might learn to choose between defensive actions.
- **Important:** The training environment is simulated. A good simulation result does not mean the system has contained a real attack.

### How the three approaches fit together

Supervised learning says, “This resembles examples labelled as an attack.” Unsupervised learning says, “This looks unusual compared with the patterns I see.” Contextual information adds facts such as vulnerability severity or asset importance. The project can combine these clues into a risk score. Reinforcement learning is a separate experiment about choosing an action in a simulated environment.

## 4. What are EDA, PCA, LDA, and SVD?

### EDA — Exploratory Data Analysis

EDA means looking through data to understand it before or while analyzing it. For example: checking how many records there are, what columns exist, whether values are missing, how many examples are labelled normal or attack, and which features appear useful.

- **Where:** Mainly Notebook **01** and the Streamlit **Data & Preprocessing Hub**.
- **Examples:** Dataset previews, class-count comparisons, feature-score displays, and plots that help inspect the data.
- **Note:** The project does not have a separate notebook named “EDA”; exploration is part of the data-preparation work.

### PCA — Principal Component Analysis

PCA turns many numeric columns into a few combined measurements. This makes complicated data easier to display on a 2D or 3D graph. It is mainly a way to summarize and visualize data; it does not itself decide that a point is an attack.

- **Where:** Notebook **01** demonstrates a PCA scatter plot; Notebook **03** uses a 2D PCA view to display anomalies. The Streamlit dimensionality-reduction graph and clustering graph also use PCA.
- **What to say about the graph:** Each dot is an example. Dots close together have similar combined feature values in this view; colors indicate the label or cluster being displayed.

### LDA — Linear Discriminant Analysis

LDA uses known class labels, such as normal and attack, to find a smaller view that separates those groups as much as possible.

- **Where:** The shared dimensionality-reduction code supports LDA when selected and supplied with labels.
- **Important:** The dashboard tab is titled “PCA/LDA/SVD,” but its current plotted dimensionality-reduction view runs **PCA**, not all three methods. Notebook 01 also calls the configured reducer and plots its output as PCA; do not claim that the current graph compares PCA, LDA, and SVD side by side.

### SVD — Singular Value Decomposition

SVD is another mathematical way to summarize a large set of features using fewer components. It is useful for working with wide or sparse data.

- **Where:** The shared dimensionality-reduction code supports SVD when selected.
- **Important:** Like LDA, SVD is not what the dashboard’s PCA-labelled graph currently plots.

**Simple memory aid:** EDA means “look at and understand the data”; PCA, LDA, and SVD are different ways to compress many measurements into fewer components. PCA is the one most visibly used for the current graphs.

## 5. What do the graphs in the Streamlit dashboard intend to show?

Streamlit is the Python dashboard. Its graphs help summarize the data, compare model results, or demonstrate how risk scoring works.

| Graph or visual | What it is intended to show |
|---|---|
| **Risk severity pie chart** | The share of analyzed results in each risk category, from minimal to critical. |
| **Signal-weight bar chart** | How much weight the system gives to supervised, anomaly, and contextual signals in the risk formula. It shows the configured weights, not the evidence for one particular event. |
| **Class-balance charts** | The number of normal and attack examples before and after balancing. The displayed “after” chart is an idealized equal-count illustration, not a plot of the resampler’s actual output. |
| **Feature-selection bar chart** | A ranking-style view of potentially useful security features. In the dashboard this chart uses example placeholder scores; it is not a measured feature-importance result. Notebook 01 separately calculates feature scores. |
| **3D PCA scatter plot** | A smaller visual view of up to 300 test examples, colored by normal/attack label. The dashboard uses PCA for this graph, even though the tab title names PCA, LDA, and SVD. |
| **Model F1 comparison chart** | Compares models using F1, a measure that balances finding attacks with avoiding false alarms. In the Streamlit lab, these are pipeline evaluation results when they are available. |
| **K-Means cluster scatter plot** | Displays groups of examples after PCA reduces the view to two dimensions. Cluster numbers are just group names; they do not mean low-to-high risk. |
| **Risk gauge and contribution pie chart** | Demonstrates how three adjustable sample scores and weights make one combined risk score. These visuals recalculate from the page’s sliders; they are a simulator, not a reading from a live event. |
| **RL reward/convergence line** | Intended to illustrate reward increasing as an agent learns. The Streamlit page currently draws a synthetic example curve with random variation, not the agent’s recorded episode-by-episode training history. |
| **Ethics robustness line chart** | If real perturbation-analysis results are available, it shows how successful evasion changes as input changes get larger. |

The Streamlit dashboard may use fallback or illustrative numbers when results are missing. In particular, the class-balance “after” chart, feature-score chart, and RL convergence curve should be introduced as demonstrations, not measured findings.

## 6. What do the graphs and visuals in the React dashboard intend to show?

React is the web dashboard connected to the project’s API. It uses more score cards, gauges, bars, and result panels than traditional scientific graphs.

| Graph or visual | What it is intended to show |
|---|---|
| **Executive severity bars and counts** | A quick view of stored events by risk category: high/critical, medium, and low/minimal. These are based on API/database responses, but event totals and risk-category counts may come from different records. |
| **Risk score gauge/bar** | Displays the combined score returned by the risk-assessment API for the sample inputs. |
| **Three evidence bars** | Breaks that combined result into supervised, anomaly, and contextual components. These are component scores—not three separate probabilities. |
| **Model benchmark cards and table** | Intended to compare example classifiers using accuracy, precision, recall, and F1. The visible benchmark figures and table are hard-coded in React, so they are not live metrics from the current training run. |
| **Anomaly score bars/cards** | Show whether each detector marks the submitted sample as unusual and the detector’s score. They use an editable example input and one API request, not a continuous sensor feed. |
| **RL status and training-result cards** | Summarize the Q-learning agent’s simulated training/evaluation, such as rewards and win rate. Some status values can be fallback display values. They describe a simulation, not real containment. |
| **Ethics, privacy, and robustness cards** | Present audit results such as fairness, privacy, and robustness. Some React display values—including the jitter/retention cards—are fixed examples or fallbacks rather than measurements returned by the audit. |
| **Recommendation cards** | Rank proposed response actions. They are suggestions; the dashboard does not actually block an IP address, quarantine a computer, or reset a password. |

The React dashboard has labels such as “LIVE,” but it does not continuously receive sensor events. Its executive page fetches a summary when opened or refreshed, and some visible incident details use example values if the API fields do not match the display.

## 7. What should I say if someone asks what the graphs prove?

> “The graphs make results easier to understand. Some show results calculated from the project pipeline or API, while others are sample simulations or display placeholders. For example, the PCA chart helps us see groups of similar examples, the risk gauge summarizes several clues, and the model chart compares model scores. The dashboards are demonstrations and analysis tools, not proof of live protection or automatic response.”

When presenting a specific number, check whether it came from a real evaluation or audit result, or whether it is a fixed example/fallback shown by the dashboard. A graph can explain what the system is designed to do, but it does not by itself prove that an attack happened or that a real system was protected.
