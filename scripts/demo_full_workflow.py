"""
End-to-End Demo of ML A/B Testing Framework

Demonstrates the complete workflow:
1. Register champion and challenger models
2. Create an A/B test experiment
3. Make predictions for various users
4. Submit ground truth feedback
5. Analyze statistical results
"""
import requests
import json
import time
import random
import pandas as pd
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80 + "\n")

def print_json(data, title=None):
    """Pretty print JSON response"""
    if title:
        print(f"\n{title}:")
    print(json.dumps(data, indent=2))

def main():
    print_section("ML A/B Testing Framework - Full Demo")
    
    # Step 1: Register Champion Model
    print_section("STEP 1: Register Champion Model")
    champion_data = {
        "model_name": "fraud_detector_v1",
        "model_type": "sklearn",
        "version": "1.0",
        "file_path": "models/fraud_detector_v1.pkl",
        "alias": "champion",
        "metadata": {
            "algorithm": "LogisticRegression",
            "accuracy": 0.937,
            "training_date": "2025-12-22"
        }
    }
    
    response = requests.post(f"{BASE_URL}/api/models/register", json=champion_data)
    champion_model = response.json()
    print_json(champion_model, "Champion Model Registered")
    champion_id = champion_model["model_id"]
    
    # Step 2: Register Challenger Model
    print_section("STEP 2: Register Challenger Model")
    challenger_data = {
        "model_name": "fraud_detector_v2",
        "model_type": "sklearn",
        "version": "2.0",
        "file_path": "models/fraud_detector_v2.pkl",
        "alias": "challenger",
        "metadata": {
            "algorithm": "RandomForestClassifier",
            "accuracy": 0.9515,
            "training_date": "2025-12-22"
        }
    }
    
    response = requests.post(f"{BASE_URL}/api/models/register", json=challenger_data)
    challenger_model = response.json()
    print_json(challenger_model, "Challenger Model Registered")
    challenger_id = challenger_model["model_id"]
    
    # Step 3: Create Experiment
    print_section("STEP 3: Create A/B Test Experiment")
    experiment_data = {
        "experiment_name": "Fraud Detection Model Comparison",
        "champion_model_id": champion_id,
        "challenger_model_id": challenger_id,
        "traffic_split": {
            "champion": 50,
            "challenger": 50
        },
        "description": "Testing if Random Forest (v2) outperforms Logistic Regression (v1)"
    }
    
    response = requests.post(f"{BASE_URL}/api/experiments/create", json=experiment_data)
    experiment = response.json()
    print_json(experiment, "Experiment Created")
    experiment_id = experiment["experiment_id"]
    
    # Step 4: Load Test Data
    print_section("STEP 4: Load Test Data for Predictions")
    test_data_path = Path("models/test_data.csv")
    test_df = pd.read_csv(test_data_path)
    print(f"Loaded {len(test_df)} test samples")
    print(f"Columns: {test_df.columns.tolist()}")
    
    # Step 5: Make Predictions
    print_section("STEP 5: Making Predictions for 100 Users")
    num_users = 100
    prediction_ids = []
    
    for user_idx in range(num_users):
        # Get random test sample
        sample_idx = random.randint(0, len(test_df) - 1)
        features = test_df.drop('is_fraud', axis=1).iloc[sample_idx].to_dict()
        ground_truth = int(test_df.iloc[sample_idx]['is_fraud'])
        
        # Make prediction
        user_id = f"user_{user_idx:04d}"
        prediction_request = {
            "user_id": user_id,
            "experiment_id": experiment_id,
            "features": features
        }
        
        response = requests.post(f"{BASE_URL}/api/predict", json=prediction_request)
        pred = response.json()
        prediction_ids.append({
            "prediction_id": pred["prediction_id"],
            "ground_truth": ground_truth,
            "user_id": user_id,
            "variant": pred["variant"]
        })
        
        if (user_idx + 1) % 20 == 0:
            print(f"Completed {user_idx + 1}/{num_users} predictions...")
    
    print(f"\nTotal predictions: {len(prediction_ids)}")
    champion_count = sum(1 for p in prediction_ids if p["variant"] == "champion")
    challenger_count = sum(1 for p in prediction_ids if p["variant"] == "challenger")
    print(f"Champion variant: {champion_count} ({champion_count/len(prediction_ids)*100:.1f}%)")
    print(f"Challenger variant: {challenger_count} ({challenger_count/len(prediction_ids)*100:.1f}%)")
    
    # Step 6: Submit Ground Truth Feedback
    print_section("STEP 6: Submitting Ground Truth Feedback")
    for idx, pred_info in enumerate(prediction_ids):
        feedback_data = {
            "prediction_id": pred_info["prediction_id"],
            "ground_truth": pred_info["ground_truth"]
        }
        
        response = requests.post(f"{BASE_URL}/api/feedback", json=feedback_data)
        
        if (idx + 1) % 20 == 0:
            print(f"Submitted feedback for {idx + 1}/{len(prediction_ids)} predictions...")
    
    print(f"\nAll {len(prediction_ids)} feedbacks submitted!")
    
    # Step 7: Get Experiment Results
    print_section("STEP 7: Statistical Analysis Results")
    print("Waiting for 2 seconds to ensure all data is processed...")
    time.sleep(2)
    
    response = requests.get(f"{BASE_URL}/api/experiments/{experiment_id}/results")
    results = response.json()
    print_json(results, "Full Results")
    
    # Step 8: Interpret Results
    print_section("STEP 8: Results Interpretation")
    
    if "error" in results:
        print(f"❌ {results['error']}")
        print(f"Champion samples: {results.get('champion_sample_size', 0)}")
        print(f"Challenger samples: {results.get('challenger_sample_size', 0)}")
        return
    
    # Sample sizes
    print(f"📊 Sample Sizes:")
    print(f"   Champion:   {results['sample_sizes']['champion']}")
    print(f"   Challenger: {results['sample_sizes']['challenger']}")
    print(f"   Total:      {results['sample_sizes']['total']}")
    
    # Performance comparison
    print(f"\n🎯 Accuracy Comparison:")
    champion_acc = results['champion']['performance']['accuracy'] * 100
    challenger_acc = results['challenger']['performance']['accuracy'] * 100
    diff = results['statistical_tests']['accuracy']['difference_percent']
    
    print(f"   Champion:   {champion_acc:.2f}%")
    print(f"   Challenger: {challenger_acc:.2f}%")
    print(f"   Difference: {diff:+.2f}%")
    
    # Statistical significance
    acc_test = results['statistical_tests']['accuracy']
    print(f"\n📈 Statistical Significance (Accuracy):")
    print(f"   Test:       {acc_test['test']}")
    print(f"   P-value:    {acc_test['p_value']:.4f}")
    print(f"   Significant: {'✅ Yes' if acc_test['statistically_significant'] else '❌ No'} (α = 0.05)")
    ci = acc_test['confidence_interval_95']
    print(f"   95% CI:     [{ci['lower_bound']:.4f}, {ci['upper_bound']:.4f}]")
    print(f"   Effect size (Cohen's d): {acc_test['effect_size_cohens_d']:.4f}")
    
    # Latency comparison
    print(f"\n⚡ Latency Comparison:")
    champion_lat = results['champion']['latency']['mean_ms']
    challenger_lat = results['challenger']['latency']['mean_ms']
    lat_diff = results['statistical_tests']['latency']['percent_change']
    
    print(f"   Champion:   {champion_lat:.2f} ms")
    print(f"   Challenger: {challenger_lat:.2f} ms")
    print(f"   Difference: {lat_diff:+.2f}%")
    
    lat_test = results['statistical_tests']['latency']
    print(f"\n   Test:       {lat_test['test']}")
    print(f"   P-value:    {lat_test['p_value']:.4f}")
    print(f"   Significant: {'✅ Yes' if lat_test['statistically_significant'] else '❌ No'} (α = 0.05)")
    
    # Recommendation
    print(f"\n💡 Recommendation:")
    rec = results['recommendation']
    print(f"   Decision:   {rec['decision'].upper()}")
    print(f"   Confidence: {rec['confidence'].upper()}")
    print(f"   Reasoning:")
    for reason in rec['reasoning']:
        print(f"      - {reason}")
    
    # Final verdict
    print_section("FINAL VERDICT")
    if rec['decision'] == 'promote_challenger':
        print("✅ PROMOTE CHALLENGER")
        print("The challenger model shows statistically significant improvements.")
        print("Safe to promote to production!")
    elif rec['decision'] == 'keep_champion':
        print("⚠️  KEEP CHAMPION")
        print("The challenger does not show significant improvements over the champion.")
        print("Continue with the current production model.")
    else:
        print("⚠️  NEEDS MANUAL REVIEW")
        print("Mixed results detected - some metrics improved, others degraded.")
        print("Manual review recommended before making a decision.")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to server at http://127.0.0.1:8000")
        print("Please ensure the server is running with: python3.11 -m uvicorn backend.main:app --reload")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
