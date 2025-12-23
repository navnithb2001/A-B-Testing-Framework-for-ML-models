"""
Quick start script - Train demo models for testing
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import joblib
import os

print("🚀 Training demo fraud detection models...\n")

# Create demo dataset (simulating fraud detection)
print("📊 Generating synthetic fraud detection dataset...")
X, y = make_classification(
    n_samples=10000,
    n_features=10,
    n_informative=8,
    n_redundant=2,
    n_classes=2,
    weights=[0.95, 0.05],  # Imbalanced: 5% fraud
    random_state=42
)

# Create feature names
feature_names = [
    'transaction_amount',
    'merchant_category',
    'user_age',
    'account_age_days',
    'num_transactions_24h',
    'avg_transaction_amount',
    'distance_from_home',
    'time_since_last_transaction',
    'device_type',
    'is_international'
]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"  • Training samples: {len(X_train)}")
print(f"  • Test samples: {len(X_test)}")
print(f"  • Fraud rate: {y.mean()*100:.1f}%\n")

# Train Model A (Champion - Logistic Regression)
print("🏆 Training Model A (Champion - Logistic Regression)...")
model_a = LogisticRegression(random_state=42, max_iter=1000)
model_a.fit(X_train, y_train)
accuracy_a = model_a.score(X_test, y_test)
print(f"  ✓ Model A accuracy: {accuracy_a*100:.2f}%")

# Save Model A
os.makedirs('models', exist_ok=True)
joblib.dump(model_a, 'models/fraud_detector_v1.pkl')
print(f"  ✓ Saved to models/fraud_detector_v1.pkl\n")

# Train Model B (Challenger - Random Forest)
print("🚀 Training Model B (Challenger - Random Forest)...")
model_b = RandomForestClassifier(n_estimators=100, random_state=42)
model_b.fit(X_train, y_train)
accuracy_b = model_b.score(X_test, y_test)
print(f"  ✓ Model B accuracy: {accuracy_b*100:.2f}%")

# Save Model B
joblib.dump(model_b, 'models/fraud_detector_v2.pkl')
print(f"  ✓ Saved to models/fraud_detector_v2.pkl\n")

# Save test data for later use
test_data = pd.DataFrame(X_test, columns=feature_names)
test_data['is_fraud'] = y_test

# Add user_id column for batch predictions
test_data.insert(0, 'user_id', [f'user_{i:04d}' for i in range(len(test_data))])

test_data.to_csv('models/test_data.csv', index=False)
print(f"  ✓ Saved test data to models/test_data.csv ({len(test_data)} rows)\n")

print("=" * 60)
print("✅ Demo models trained successfully!")
print("=" * 60)
print(f"\nModel Comparison:")
print(f"  Champion (Logistic Regression): {accuracy_a*100:.2f}% accuracy")
print(f"  Challenger (Random Forest):     {accuracy_b*100:.2f}% accuracy")
print(f"  Improvement: {(accuracy_b - accuracy_a)*100:+.2f} percentage points")
print("\n💡 Next steps:")
print("  1. Start the API: uvicorn backend.main:app --reload")
print("  2. Register both models via API")
print("  3. Create an A/B test experiment")
print("  4. Start making predictions!")
print("\n" + "=" * 60)
