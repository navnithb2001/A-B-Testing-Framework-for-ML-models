# ML Model A/B Testing Framework 🚀

> A production-ready system for comparing machine learning models in live environments with rigorous statistical analysis.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Using Your Own Models](#-using-your-own-models)
- [API Documentation](#-api-documentation)
- [Statistical Analysis](#-statistical-analysis)
- [Project Structure](#-project-structure)
- [Development](#-development)
- [Demo Workflow](#-demo-workflow)

---

## 🎯 Overview

This framework enables **data scientists and ML engineers** to:

✅ Deploy multiple model versions simultaneously (champion vs challenger)  
✅ Route production traffic intelligently using consistent hashing  
✅ Collect predictions and ground truth feedback  
✅ Calculate comprehensive performance metrics  
✅ Perform rigorous statistical hypothesis testing  
✅ Make automated, data-driven deployment decisions  

**Perfect for:** Model validation, online A/B testing, gradual rollouts, performance monitoring

---

## ✨ Features

### 🔄 Model Management
- Register multiple model versions with metadata
- Version control and artifact storage
- Support for sklearn, XGBoost, LightGBM, and custom models
- Hot-swapping models without downtime

### 🎲 Traffic Routing
- **Consistent hashing** for deterministic user assignment
- Configurable traffic splits (e.g., 80/20, 50/50)
- Minimal user reassignment when experiments change
- Per-user variant persistence across sessions

### 📊 Metrics & Analytics
- **Classification:** Accuracy, Precision, Recall, F1, AUC-ROC, Confusion Matrix
- **Regression:** MSE, RMSE, MAE, R², MAPE
- **Latency:** Mean, Median, p50, p95, p99
- **Custom metrics** support
- **Batch predictions** - Upload CSV files for bulk testing

### 📈 Statistical Analysis (⭐ Most Impressive)
- **Welch's t-test** - Latency comparison with unequal variance handling
- **Two-proportion z-test** - Accuracy/conversion rate comparison
- **Confidence intervals (95%)** - Precision estimation for differences
- **Effect size (Cohen's d)** - Practical significance measurement
- **Automated recommendations** - Promote, keep, or review decisions
- **Multi-factor reasoning** - Balances accuracy vs latency trade-offs

### 🎛️ Production Ready
- RESTful API with FastAPI
- SQLite database (easily upgradable to PostgreSQL)
- Comprehensive error handling
- Detailed logging
- Auto-generated interactive API docs

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Production Traffic                      │
└────────────────────────────┬────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   FastAPI App   │
                    │  (Port 8000)    │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
    ┌───▼────┐         ┌─────▼──────┐       ┌─────▼──────┐
    │ Models │         │Experiments │       │Predictions │
    │Registry│         │ Manager    │       │  Router    │
    └───┬────┘         └─────┬──────┘       └─────┬──────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   SQLite DB     │
                    │ (3 tables)      │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────▼─────┐      ┌───────▼────────┐    ┌──────▼──────┐
   │Champion  │      │   Challenger   │    │  Feedback   │
   │Model.pkl │      │   Model.pkl    │    │ Collection  │
   └──────────┘      └────────────────┘    └──────┬──────┘
                                                  │
                                          ┌───────▼────────┐
                                          │   Statistical  │
                                          │    Analysis    │
                                          └────────────────┘
```

**Data Flow:**
1. User request → Traffic router assigns variant (champion/challenger)
2. Load appropriate model from disk
3. Make prediction, log to database with latency tracking
4. Collect ground truth feedback asynchronously
5. Statistical engine analyzes results when requested
6. Automated recommendation (promote/keep/review)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip
- 2GB free disk space

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/navnithb2001/AutoApply.git
cd AutoApply

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
cp .env.example .env

# 4. Train demo models (optional - for testing)
python3.11 scripts/train_demo_models.py

# 5. Start the server
python3.11 -m uvicorn backend.main:app --reload
```

**Server will start at:** http://127.0.0.1:8000  
**Interactive API docs:** http://127.0.0.1:8000/docs  

---

## 🎮 Demo Workflow

Run the complete end-to-end demo:

```bash
# Make sure server is running in background
python3.11 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 &

# Wait for startup
sleep 3

# Run the full demo
python3.11 scripts/demo_full_workflow.py
```

**The demo will:**
1. ✅ Register champion model (Logistic Regression, 93.70% accuracy)
2. ✅ Register challenger model (Random Forest, 95.15% accuracy)
3. ✅ Create A/B test experiment (50/50 traffic split)
4. ✅ Make 100 predictions across both variants
5. ✅ Submit ground truth feedback for all predictions
6. ✅ Generate comprehensive statistical analysis
7. ✅ Provide clear recommendation (promote/keep/review)

**Expected Output:**
```
📊 Sample Sizes: 47 champion, 53 challenger
🎯 Accuracy: Champion 95.74%, Challenger 96.23% (+0.48%)
📈 P-value: 0.9023 (❌ Not significant)
⚡ Latency: Champion 0.05ms, Challenger 1.13ms (+2229.67%)
💡 Recommendation: KEEP_CHAMPION (High Confidence)
   Reasoning:
   - No significant difference in accuracy detected
   - Challenger is significantly slower (+2229.67%)
```

---

## 🔧 Using Your Own Models

### Option 1: Use Existing Trained Models

```python
# 1. Save your models with joblib
import joblib

joblib.dump(your_model_v1, 'models/my_champion.pkl')
joblib.dump(your_model_v2, 'models/my_challenger.pkl')

# 2. Register via API
curl -X POST http://127.0.0.1:8000/api/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "my_model",
    "version": "v1",
    "file_path": "models/my_champion.pkl",
    "model_type": "sklearn",
    "alias": "champion",
    "metadata": {"algorithm": "RandomForest", "accuracy": 0.95}
  }'
```

### Option 2: Train Models from Scratch

Use the provided template:

```bash
# 1. Copy the template
cp scripts/train_your_models_template.py scripts/train_my_models.py

# 2. Edit the script - change these sections:
#    - Load your dataset (line 27)
#    - Set your target column (line 40)
#    - Choose champion algorithm (line 54)
#    - Choose challenger algorithm (line 74)

# 3. Run training
python3.11 scripts/train_my_models.py

# 4. Models will be saved to models/
```

### Supported Model Types

✅ **Scikit-learn:** `LogisticRegression`, `RandomForest`, `GradientBoosting`, `SVC`, etc.  
✅ **XGBoost:** `XGBClassifier`, `XGBRegressor`  
✅ **LightGBM:** `LGBMClassifier`, `LGBMRegressor`  
✅ **Any model** with `.predict()` method that can be saved with joblib

### Complete Example: Customer Churn Prediction

```python
"""Train custom models for comparing churn prediction algorithms"""
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# 1. Load your data
df = pd.read_csv('customer_data.csv')
X = df.drop('churned', axis=1)
y = df['churned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Train champion (current production model)
champion = RandomForestClassifier(n_estimators=50, random_state=42)
champion.fit(X_train, y_train)
joblib.dump(champion, 'models/churn_v1.pkl')
print(f"Champion accuracy: {champion.score(X_test, y_test)*100:.2f}%")

# 3. Train challenger (new model to test)
challenger = GradientBoostingClassifier(n_estimators=100, random_state=42)
challenger.fit(X_train, y_train)
joblib.dump(challenger, 'models/churn_v2.pkl')
print(f"Challenger accuracy: {challenger.score(X_test, y_test)*100:.2f}%")

# 4. Save test data for predictions
test_data = X_test.copy()
test_data['churned'] = y_test
test_data.to_csv('models/churn_test_data.csv', index=False)

print("\n✅ Models saved! Now register them via API and create an experiment.")
```

### Best Practices

1. **Feature Consistency:** Both models must expect the same features in the same order
2. **Test Data:** Always save test data matching your model's feature set
3. **Sample Size:** Collect at least 30+ predictions per variant for valid statistics
4. **Experiment Duration:** Run experiments for meaningful time periods (days/weeks)

---

## 📚 API Documentation

### Model Management

#### Register a Model
```bash
POST /api/models/register

{
  "model_name": "fraud_detector",
  "version": "v1",
  "file_path": "models/fraud_detector_v1.pkl",
  "model_type": "sklearn",
  "alias": "champion",
  "metadata": {"algorithm": "LogisticRegression"}
}

Response: {"status": "success", "model_id": "model_abc123"}
```

#### List All Models
```bash
GET /api/models/list

Response: [
  {"id": "model_abc123", "model_name": "fraud_detector", "version": "v1", ...},
  ...
]
```

#### Get Model Details
```bash
GET /api/models/{model_id}
```

#### Delete Model
```bash
DELETE /api/models/{model_id}

Response: {"status": "success", "message": "Model model_abc123 deleted"}
```

**Note:** Models cannot be deleted if they are used in any experiments. Delete experiments first.

**Error Example:**
```json
{
  "detail": "Cannot delete model. It is used in 2 experiment(s). 
             Please delete the experiments first or stop using this model."
}
```

### Experiment Management

#### Create A/B Test
```bash
POST /api/experiments/create

{
  "experiment_name": "Fraud Detection Comparison",
  "champion_model_id": "model_abc123",
  "challenger_model_id": "model_def456",
  "traffic_split": {"champion": 50, "challenger": 50},
  "primary_metric": "accuracy",
  "duration_days": 7,
  "description": "Testing Random Forest vs Logistic Regression"
}

Response: {"status": "success", "experiment_id": "exp_xyz789"}
```

#### List Experiments
```bash
GET /api/experiments/list?status=running
```

#### Stop Experiment
```bash
POST /api/experiments/{experiment_id}/stop

Response: {
  "status": "success",
  "message": "Experiment exp_xyz789 stopped"
}
```

#### Delete Experiment
```bash
DELETE /api/experiments/{experiment_id}

Response: {
  "status": "success",
  "message": "Experiment exp_xyz789 deleted",
  "predictions_deleted": 1250
}
```

**Note:** Deleting an experiment will also delete all associated predictions. This action is permanent.

#### Get Experiment Results (⭐ Statistical Analysis)
```bash
GET /api/experiments/{experiment_id}/results

Response: {
  "experiment_id": "exp_xyz789",
  "status": "running",
  "sample_sizes": {"champion": 1250, "challenger": 1250},
  "champion": {
    "performance": {"accuracy": 0.94, "precision": 0.92, ...},
    "latency": {"mean_ms": 12.5, "p95_ms": 18.3, ...}
  },
  "challenger": {
    "performance": {"accuracy": 0.96, "precision": 0.95, ...},
    "latency": {"mean_ms": 45.2, "p95_ms": 67.8, ...}
  },
  "statistical_tests": {
    "accuracy": {
      "difference_percent": 2.13,
      "p_value": 0.003,
      "statistically_significant": true,
      "confidence_interval_95": {"lower_bound": 0.007, "upper_bound": 0.036},
      "effect_size_cohens_d": 0.42
    },
    "latency": {
      "percent_change": 261.6,
      "p_value": 0.0001,
      "statistically_significant": true
    }
  },
  "recommendation": {
    "decision": "needs_review",
    "reasoning": [
      "Challenger has significantly better accuracy (+2.13%)",
      "Challenger is significantly slower (+261.6%)"
    ],
    "confidence": "high"
  }
}
```

### Prediction & Feedback

#### Make Prediction
```bash
POST /api/predict

{
  "experiment_id": "exp_xyz789",
  "user_id": "user_12345",
  "features": {
    "transaction_amount": 1250.50,
    "merchant_category": 2,
    "user_age": 35,
    ...
  }
}

Response: {
  "prediction": 0,
  "variant": "challenger",
  "prediction_id": "pred_abc123",
  "latency_ms": 15.2
}
```

#### Batch Prediction (Upload CSV)
```bash
POST /api/batch-predict

# Using curl with file upload
curl -X POST http://127.0.0.1:8000/api/batch-predict \
  -F "file=@test_data.csv" \
  -F "experiment_id=exp_xyz789" \
  -F "user_id_column=user_id" \
  -F "target_column=is_fraud"

Response: {
  "status": "success",
  "total_rows": 2000,
  "successful_predictions": 2000,
  "predictions_by_variant": {"champion": 1004, "challenger": 996},
  "overall_accuracy": 0.944,
  "results": [...first 100 predictions...],
  "message": "Processed 2000 rows. Use /api/experiments/{id}/results for analysis."
}
```

**CSV Format Requirements:**
- Must include a column for user IDs (default: `user_id`)
- Feature columns should match model training data
- Optionally include ground truth column for automatic feedback
- Example CSV structure:
  ```csv
  user_id,transaction_amount,merchant_category,...,is_fraud
  user_0001,1250.50,2,...,0
  user_0002,850.20,1,...,1
  ```

#### Submit Ground Truth Feedback
```bash
POST /api/feedback

{
  "prediction_id": "pred_abc123",
  "ground_truth": 1
}

Response: {"status": "success"}
```

---

## 📊 Statistical Analysis Guide

### Understanding the Results

When you get experiment results, here's how to interpret them:

#### P-value
- **< 0.05:** Statistically significant difference (reject null hypothesis)
- **> 0.05:** No significant difference detected
- **Lower is stronger** evidence of real difference

#### Confidence Interval (95% CI)
- Range where the **true difference likely lies**
- **If CI contains 0:** No significant difference
- **Narrower CI:** More precise estimate

#### Effect Size (Cohen's d)
- **< 0.2:** Small effect
- **0.2 - 0.5:** Medium effect  
- **> 0.5:** Large effect
- Measures **practical** vs **statistical** significance

### Decision Guidelines

#### ✅ Promote Challenger If:
- Primary metric (accuracy) **significantly better** (p < 0.05)
- Latency **acceptable** (not significantly worse)
- Effect size **meaningful** (d > 0.2)
- Sample size **adequate** (n > 30 per variant)

#### ⚠️ Keep Champion If:
- **No significant** accuracy improvement
- Latency **significantly worse**
- Effect size **too small** (d < 0.1)
- **Insufficient data** collected

#### 🔍 Needs Review If:
- **Mixed results** (accuracy up but latency down)
- **Borderline significance** (0.05 < p < 0.10)
- Very **large confidence intervals**
- **Edge cases** or anomalies detected

### Example Interpretation

```json
{
  "accuracy": {
    "difference_percent": 1.45,
    "p_value": 0.023,
    "confidence_interval_95": {"lower_bound": 0.002, "upper_bound": 0.029},
    "effect_size_cohens_d": 0.35
  }
}
```

**Interpretation:**
- ✅ Accuracy improved by **1.45%**
- ✅ **p=0.023 < 0.05** → Statistically significant
- ✅ **CI [0.002, 0.029]** doesn't contain 0 → Confirms real improvement
- ✅ **Effect size 0.35** is medium → Practically meaningful
- ✅ **Recommendation:** Promote challenger

---

## 📁 Project Structure

```
AutoApply/
├── backend/
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Configuration management
│   ├── database.py             # SQLAlchemy ORM models
│   ├── routers/
│   │   ├── models.py           # Model registry endpoints
│   │   ├── experiments.py      # Experiment management
│   │   └── predictions.py      # Prediction serving & feedback
│   └── statistical/
│       ├── tests.py            # Statistical hypothesis tests
│       └── metrics.py          # Performance metrics calculator
├── models/
│   ├── fraud_detector_v1.pkl   # Demo champion model
│   ├── fraud_detector_v2.pkl   # Demo challenger model
│   └── test_data.csv           # Demo test dataset
├── scripts/
│   ├── train_demo_models.py    # Generate demo models
│   ├── demo_full_workflow.py   # End-to-end demo
│   └── train_your_models_template.py  # Template for custom models
├── .env.example                # Environment variables template
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── ml_ab_testing.db           # SQLite database (auto-created)
```

---

## 🛠️ Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests with coverage
pytest tests/ --cov=backend --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Code Quality

```bash
# Format code
black backend/ scripts/

# Lint code
pylint backend/ scripts/

# Type checking
mypy backend/
```

### Database Migrations

```bash
# If you modify database.py models:
# 1. Delete the existing database
rm ml_ab_testing.db

# 2. Restart the server (auto-creates new schema)
python3.11 -m uvicorn backend.main:app --reload
```

---

## 🔮 Future Enhancements

- [ ] **React Dashboard** - Interactive visualizations and monitoring
- [ ] **PostgreSQL Support** - Production database backend
- [ ] **Docker Deployment** - Containerization with docker-compose
- [ ] **Multi-armed Bandits** - Adaptive traffic allocation
- [ ] **Bayesian A/B Testing** - Alternative to frequentist approach
- [ ] **Feature Flagging** - Integration with LaunchDarkly/Flagsmith
- [ ] **Slack/Email Alerts** - Automated notifications
- [ ] **Cost Tracking** - Monitor inference costs per variant
- [ ] **Model Explainability** - SHAP/LIME integration
- [ ] **Time Series Analysis** - Trend detection over time

---

## 📖 Additional Resources

- **Interactive API Docs:** http://127.0.0.1:8000/docs (when server running)
- **OpenAPI Spec:** http://127.0.0.1:8000/openapi.json
- **Redoc:** http://127.0.0.1:8000/redoc

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [SQLAlchemy](https://www.sqlalchemy.org/) - SQL toolkit and ORM
- [Scikit-learn](https://scikit-learn.org/) - Machine learning library
- [SciPy](https://scipy.org/) - Scientific computing
- [Pydantic](https://pydantic-docs.helpmanual.io/) - Data validation

---

## 📧 Contact

**Navnith Bharadwaj**  
GitHub: [@navnithb2001](https://github.com/navnithb2001)  
Project Link: [https://github.com/navnithb2001/A-B-Testing-Framework-for-ML-models](https://github.com/navnithb2001/A-B-Testing-Framework-for-ML-models)

---

## ⭐ Star History

If this project helped you, please consider giving it a star! ⭐
