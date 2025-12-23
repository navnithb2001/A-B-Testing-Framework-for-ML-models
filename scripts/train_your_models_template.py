"""
Template: Train and compare your own models

This is a template you can copy and modify for your specific use case.
Just replace the dataset loading and model configurations with your own.
"""
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ============================================================================
# STEP 1: Load YOUR data
# ============================================================================
print("📊 Loading dataset...\n")

# OPTION A: Load from CSV
# df = pd.read_csv('path/to/your/data.csv')

# OPTION B: Load from database
# import sqlalchemy
# engine = sqlalchemy.create_engine('your_connection_string')
# df = pd.read_sql('SELECT * FROM your_table', engine)

# OPTION C: Use sklearn sample dataset (for testing this template)
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(f"Dataset shape: {df.shape}")
print(f"Features: {df.columns.tolist()[:5]}... ({len(df.columns)-1} total)")
print(f"Target distribution:\n{df['target'].value_counts()}\n")

# ============================================================================
# STEP 2: Prepare features and target
# ============================================================================
print("🔧 Preparing features and target...\n")

# TODO: Change 'target' to your actual target column name
TARGET_COLUMN = 'target'  # <-- CHANGE THIS

X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Number of features: {X.shape[1]}\n")

# ============================================================================
# STEP 3: Train Champion Model (Your current production model)
# ============================================================================
print("🏆 Training Champion Model...\n")

# TODO: Replace with your current production model
from sklearn.linear_model import LogisticRegression

champion = LogisticRegression(
    random_state=42,
    max_iter=1000,
    # Add your hyperparameters here
)

champion.fit(X_train, y_train)
champion_pred = champion.predict(X_test)
champion_accuracy = accuracy_score(y_test, champion_pred)

print(f"Champion Model: LogisticRegression")
print(f"Accuracy: {champion_accuracy*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, champion_pred))

# Save champion model
os.makedirs('models', exist_ok=True)
CHAMPION_PATH = 'models/my_champion.pkl'  # <-- Change filename if desired
joblib.dump(champion, CHAMPION_PATH)
print(f"✅ Saved champion to {CHAMPION_PATH}\n")

# ============================================================================
# STEP 4: Train Challenger Model (New model you want to test)
# ============================================================================
print("🚀 Training Challenger Model...\n")

# TODO: Replace with your new model you want to test
from sklearn.ensemble import RandomForestClassifier

challenger = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    # Add your hyperparameters here
)

challenger.fit(X_train, y_train)
challenger_pred = challenger.predict(X_test)
challenger_accuracy = accuracy_score(y_test, challenger_pred)

print(f"Challenger Model: RandomForestClassifier")
print(f"Accuracy: {challenger_accuracy*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, challenger_pred))

# Save challenger model
CHALLENGER_PATH = 'models/my_challenger.pkl'  # <-- Change filename if desired
joblib.dump(challenger, CHALLENGER_PATH)
print(f"✅ Saved challenger to {CHALLENGER_PATH}\n")

# ============================================================================
# STEP 5: Save test data for later predictions
# ============================================================================
print("💾 Saving test data...\n")

test_data = X_test.copy()
test_data[TARGET_COLUMN] = y_test
TEST_DATA_PATH = 'models/my_test_data.csv'  # <-- Change filename if desired
test_data.to_csv(TEST_DATA_PATH, index=False)
print(f"✅ Saved test data to {TEST_DATA_PATH}\n")

# ============================================================================
# STEP 6: Summary and Next Steps
# ============================================================================
print("=" * 70)
print("✅ MODELS TRAINED SUCCESSFULLY!")
print("=" * 70)
print(f"\n📊 Results Summary:")
print(f"  Champion (LogisticRegression): {champion_accuracy*100:.2f}% accuracy")
print(f"  Challenger (RandomForest):     {challenger_accuracy*100:.2f}% accuracy")
print(f"  Difference:                    {(challenger_accuracy - champion_accuracy)*100:+.2f}%")

if challenger_accuracy > champion_accuracy:
    print(f"\n🎯 Challenger looks promising! (+{(challenger_accuracy - champion_accuracy)*100:.2f}%)")
    print("   But is it STATISTICALLY SIGNIFICANT? Let's A/B test it!")
else:
    print(f"\n⚠️  Challenger didn't improve offline accuracy.")
    print("   But maybe it's faster? Or better on specific segments?")
    print("   Let's A/B test to find out!")

print("\n" + "=" * 70)
print("📝 NEXT STEPS:")
print("=" * 70)
print("""
1. Start the API server:
   python3.11 -m uvicorn backend.main:app --reload

2. Register Champion Model:
   curl -X POST http://127.0.0.1:8000/api/models/register \\
     -H "Content-Type: application/json" \\
     -d '{
       "model_name": "my_model",
       "version": "v1",
       "file_path": "models/my_champion.pkl",
       "model_type": "sklearn",
       "alias": "champion",
       "metadata": {"algorithm": "LogisticRegression"}
     }'

3. Register Challenger Model:
   curl -X POST http://127.0.0.1:8000/api/models/register \\
     -H "Content-Type: application/json" \\
     -d '{
       "model_name": "my_model",
       "version": "v2",
       "file_path": "models/my_challenger.pkl",
       "model_type": "sklearn",
       "alias": "challenger",
       "metadata": {"algorithm": "RandomForest"}
     }'

4. Create A/B Test Experiment:
   curl -X POST http://127.0.0.1:8000/api/experiments/create \\
     -H "Content-Type: application/json" \\
     -d '{
       "experiment_name": "My Model Comparison",
       "champion_model_id": "model_XXXXX",
       "challenger_model_id": "model_YYYYY",
       "traffic_split": {"champion": 50, "challenger": 50},
       "primary_metric": "accuracy",
       "duration_days": 7
     }'

5. Make predictions and collect feedback (see demo_full_workflow.py)

6. Analyze results:
   curl http://127.0.0.1:8000/api/experiments/exp_ZZZZZ/results

For full examples, see:
- scripts/demo_full_workflow.py (complete end-to-end workflow)
- USING_YOUR_OWN_MODELS.md (detailed guide)
""")

# ============================================================================
# OPTIONAL: Quick model verification
# ============================================================================
print("\n" + "=" * 70)
print("🔍 MODEL VERIFICATION:")
print("=" * 70)

# Test that models can be loaded and predict
print("\nTesting model loading and prediction...")

loaded_champion = joblib.load(CHAMPION_PATH)
loaded_challenger = joblib.load(CHALLENGER_PATH)

# Get a sample from test data
sample_features = X_test.iloc[0:1]
print(f"\nSample features shape: {sample_features.shape}")

# Test predictions
champ_pred = loaded_champion.predict(sample_features)
chall_pred = loaded_challenger.predict(sample_features)

print(f"Champion prediction: {champ_pred[0]}")
print(f"Challenger prediction: {chall_pred[0]}")
print(f"Actual value: {y_test.iloc[0]}")

print("\n✅ Models can be loaded and predict successfully!")
print("\nYou're ready to run A/B tests! 🚀\n")
