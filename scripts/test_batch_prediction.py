"""
Test batch prediction endpoint with CSV upload
"""
import requests
import time

API_URL = "http://127.0.0.1:8000"

print("🧪 Testing Batch Prediction Endpoint\n")

# Step 1: Register models
print("1️⃣  Registering models...")
champion = requests.post(f"{API_URL}/api/models/register", json={
    "model_name": "fraud_detector",
    "version": "v1",
    "file_path": "models/fraud_detector_v1.pkl",
    "model_type": "sklearn",
    "alias": "champion"
})
champion_id = champion.json()["model_id"]
print(f"   ✅ Champion registered: {champion_id}")

challenger = requests.post(f"{API_URL}/api/models/register", json={
    "model_name": "fraud_detector",
    "version": "v2",
    "file_path": "models/fraud_detector_v2.pkl",
    "model_type": "sklearn",
    "alias": "challenger"
})
challenger_id = challenger.json()["model_id"]
print(f"   ✅ Challenger registered: {challenger_id}")

# Step 2: Create experiment
print("\n2️⃣  Creating experiment...")
exp = requests.post(f"{API_URL}/api/experiments/create", json={
    "experiment_name": "Batch CSV Test",
    "champion_model_id": champion_id,
    "challenger_model_id": challenger_id,
    "traffic_split": {"champion": 50, "challenger": 50},
    "primary_metric": "accuracy",
    "duration_days": 7
})
exp_id = exp.json()["experiment_id"]
print(f"   ✅ Experiment created: {exp_id}")

# Step 3: Upload CSV for batch prediction
print("\n3️⃣  Uploading CSV file for batch predictions...")
time.sleep(1)  # Give server time to process

with open('models/test_data.csv', 'rb') as f:
    files = {'file': ('test_data.csv', f, 'text/csv')}
    data = {
        'experiment_id': exp_id,
        'user_id_column': 'user_id',
        'target_column': 'is_fraud'
    }
    
    response = requests.post(
        f"{API_URL}/api/batch-predict",
        files=files,
        data=data
    )

if response.status_code == 200:
    result = response.json()
    print(f"\n   ✅ Batch prediction successful!")
    print(f"\n   📊 Summary:")
    print(f"      • Total rows processed: {result['total_rows']}")
    print(f"      • Successful predictions: {result['successful_predictions']}")
    print(f"      • Failed predictions: {result['failed_predictions']}")
    print(f"      • Champion predictions: {result['predictions_by_variant']['champion']}")
    print(f"      • Challenger predictions: {result['predictions_by_variant']['challenger']}")
    
    if result['overall_accuracy']:
        print(f"      • Overall accuracy: {result['overall_accuracy']*100:.2f}%")
    
    print(f"\n   🔍 First 5 predictions:")
    for i, pred in enumerate(result['results'][:5], 1):
        correct_symbol = "✅" if pred.get('correct') else "❌"
        print(f"      {i}. User {pred['user_id']}: {pred['variant']} → {pred['prediction']} "
              f"(truth: {pred.get('ground_truth', 'N/A')}) {correct_symbol}")
else:
    print(f"\n   ❌ Error: {response.status_code}")
    print(f"      {response.text}")

# Step 4: Get experiment results
print("\n4️⃣  Getting experiment statistical results...")
time.sleep(1)

results = requests.get(f"{API_URL}/api/experiments/{exp_id}/results")

if results.status_code == 200:
    stats = results.json()
    print(f"\n   ✅ Statistical Analysis:")
    print(f"      • Champion samples: {stats['sample_sizes']['champion']}")
    print(f"      • Challenger samples: {stats['sample_sizes']['challenger']}")
    
    if 'champion' in stats and 'performance' in stats['champion']:
        champ_acc = stats['champion']['performance'].get('accuracy', 0)
        chal_acc = stats['challenger']['performance'].get('accuracy', 0)
        print(f"      • Champion accuracy: {champ_acc*100:.2f}%")
        print(f"      • Challenger accuracy: {chal_acc*100:.2f}%")
    
    if 'recommendation' in stats:
        print(f"\n   💡 Recommendation: {stats['recommendation']['decision'].upper()}")
        if 'reasoning' in stats['recommendation']:
            for reason in stats['recommendation']['reasoning']:
                print(f"      - {reason}")
else:
    print(f"   ⚠️  Not enough data yet for statistical analysis")

print("\n✅ Test complete!")
print(f"\n💡 Tip: You can now view detailed results at:")
print(f"   http://127.0.0.1:8000/api/experiments/{exp_id}/results")
