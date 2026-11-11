SpamGuard ML — Email Spam Prediction System
An interactive machine learning email spam classification system powered by Multinomial Naive Bayes (MNB), Linear Support Vector Machines (Linear SVM), and Complement Naive Bayes (CNB) with TF-IDF vectorization, Explainable AI (XAI) token attribution, and comprehensive model evaluation metrics.
📌 Project Overview
SpamGuard ML is a full-stack, client-side machine learning application that detects spam, phishing, and scam emails from raw text in real-time (sub-millisecond latency). The mathematical algorithms are implemented from scratch in TypeScript, providing deep visibility into probability distributions, decision hyperplanes, and token-level feature importance without relying on black-box external APIs.
🚀 Key Features
Multiple ML Classifiers (Side-by-Side Comparison):
Multinomial Naive Bayes (MNB) with Laplace smoothing (
) and log-space underflow protection.
Linear Support Vector Machine (Linear SVM) with soft-margin hinge loss optimization via the Pegasos Stochastic Subgradient Descent (SGD) algorithm.
Complement Naive Bayes (CNB) tailored for class-imbalanced text corpora.
Consensus Mode: Runs predictions across all three models simultaneously.
Explainable AI (XAI) & Token Attribution:
Interactive visual word highlighter: Tokens are colored by their log-odds or SVM weight (red for spam indicators, green for legitimate/ham indicators).
Hover inspection: Inspect exact token frequency, log-odds ratio, and weight contribution.
Feature Importance Bar Chart: Visualizes the top 6 spam vs. ham feature drivers.
Structural Heuristics: Monitors ALL-CAPS ratios, exclamation marks (!), currency symbols ($), and urgent trigger phrases.
Model Lab & Evaluation Suite:
Stratified train/test holdout evaluation (Accuracy, Precision, Recall, F1-Score, Training Latency).
Interactive Confusion Matrix: Real-time True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN). Click any misclassification to inspect the exact email sample.
Receiver Operating Characteristic (ROC Curve): Interactive SVG curve plotting True Positive Rate vs. False Positive Rate across 20+ thresholds with calculated AUC (Area Under Curve).
Hyperparameter Tuning Lab: Adjust Laplace smoothing 
, SVM regularization 
, SGD epochs, N-grams (unigrams vs. bigrams), and vectorization schemes (TF-IDF vs. BoW vs. Binary).
Corpus & Dataset Explorer:
Pre-loaded with 70 authentic benchmark emails spanning phishing, cryptocurrency scams, advance-fee fraud, counterfeit goods, work collaboration, transactional receipts, and security alerts.
Search by keyword, filter by category or label, add custom samples, export/import as CSV/JSON, and reset to benchmark anytime.
Batch Classifier & Theory Guide:
Bulk classification of multiple emails or subjects with CSV report download.
Mathematical theory guide detailing Bayes' Theorem, maximum-margin hyperplanes, hinge loss subgradients, and Platt scaling.
🧮 Machine Learning Formulations
1. Multinomial Naive Bayes (MNB)
Based on Bayes' Theorem under the conditional independence assumption:
To prevent floating-point underflow with long documents, calculations are performed in log-space:
With Laplace Additive Smoothing (
):
2. Linear Support Vector Machine (Pegasos SGD)
Minimizes the soft-margin primal objective:

where 
 and 
 balances margin width against classification error.
Subgradient weight update for margin violations (
):
Calibrated probability via Platt Scaling:
3. TF-IDF Vectorization
Term Frequency (TF): 
Inverse Document Frequency (IDF): 
Vector Transformation: 
 normalized to unit Euclidean length (
 norm).
📂 Project Structure
Code
├── index.html                      # HTML entry point with fonts & metadata
├── metadata.json                   # AI Studio applet metadata
├── package.json                    # Project dependencies and npm scripts
├── tsconfig.json                   # TypeScript configuration
├── vite.config.ts                  # Vite build and server configuration
├── src/
│   ├── main.tsx                    # React DOM root entry point
│   ├── App.tsx                     # Main application layout, routing & state
│   ├── index.css                   # Tailwind CSS styling and theme setup
│   ├── components/
│   │   ├── Header.tsx              # Clean top navigation bar (Top Bar Contract)
│   │   ├── PredictorView.tsx       # Live playground with presets & XAI highlighter
│   │   ├── ModelLabView.tsx        # Holdout metrics, Confusion Matrix & ROC Curve
│   │   ├── DatasetView.tsx         # Searchable table, dataset management & CSV import/export
│   │   ├── BatchTesterView.tsx     # High-throughput batch classification tool
│   │   └── AlgorithmGuideView.tsx  # Mathematical formulations and derivations
│   └── ml/
│       ├── types.ts                # TypeScript interfaces (metrics, samples, hyperparameters)
│       ├── stopwords.ts            # English NLP stopword vocabulary
│       ├── tokenizer.ts            # N-gram generation, punctuation & heuristic signals
│       ├── vectorizer.ts           # TF-IDF, Bag-of-Words & L2 normalization
│       ├── naiveBayes.ts           # Multinomial & Complement Naive Bayes implementation
│       ├── svm.ts                  # Linear SVM with subgradient SGD & Platt scaling
│       ├── evaluator.ts            # Stratified train/test split, Confusion Matrix & ROC AUC
│       ├── dataset.ts              # 70 benchmark emails + local storage persistence
│       └── modelManager.ts         # Unified controller orchestrating training & inferenc
