import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
from datetime import datetime

train_df = pd.read_parquet('data/train_tabular.parquet')
test_df = pd.read_parquet('data/test_tabular.parquet')
train_data_X = train_df[['room_type','neighbourhood_cleansed','accommodates','bathrooms','bedrooms','minimum_nights']]
train_data_Y = train_df[['price']].values.ravel()
test_data_X = test_df[['room_type','neighbourhood_cleansed','accommodates','bathrooms','bedrooms','minimum_nights']]
test_data_Y = test_df[['price']].values.ravel()


def train_and_evaluate(X_train, y_train, X_test, y_test):
    best_model = None
    best_mse = float('inf')
    best_mae = float('inf')
    best_r2 = float('-inf')
    best_params_dict = {}
    all_results = []
    
    max_depths = [8, 12, 15, 20, 25, 30]
    min_samples_leafs = [5, 10, 20, 30]
    
    print("Training models with different hyperparameters...")
    print("-" * 70)
    print(f"{'max_depth':<12} {'min_samples_leaf':<18} {'RMSE':<12} {'MAE':<12} {'R2':<10}")
    print("-" * 70)
    
    for max_depth in max_depths:
        for min_samples_leaf in min_samples_leafs:
            model = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            all_results.append({'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf, 
                               'rmse': rmse, 'mae': mae, 'r2': r2})
            
            print(f"{max_depth:<12} {min_samples_leaf:<18} {rmse:<12.4f} {mae:<12.4f} {r2:<10.4f}")
            
            if rmse < best_mse:
                best_mse = rmse
                best_mae = mae
                best_r2 = r2
                best_model = model
                best_params_dict = {'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf}
    
    print("-" * 70)
    
    return best_model, best_params_dict, {'rmse': best_mse, 'mae': best_mae, 'r2': best_r2}, all_results


best_model, best_params, best_metrics, all_results = train_and_evaluate(train_data_X, train_data_Y, test_data_X, test_data_Y)

print("\nTraining complete!")
print("\nBest DecisionTreeRegressor hyperparameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")
print("\nBest model test set metrics:")
for metric, value in best_metrics.items():
    print(f"  {metric.upper()}: {value:.4f}")

y_pred_best = best_model.predict(test_data_X)

print("\nSample predictions vs actuals:")
for i in range(min(5, len(test_data_Y))):
    print(f"  Actual: ${test_data_Y[i]:.2f}, Predicted: ${y_pred_best[i]:.2f}")

feature_importances = best_model.feature_importances_
feature_names = ['room_type','neighbourhood_cleansed','accommodates','bathrooms','bedrooms','minimum_nights']

print("\nFeature importances:")
for name, importance in sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {importance:.4f}")

output_dir = Path('outputs')
output_dir.mkdir(exist_ok=True)
csv_path = output_dir / 'model_runs.csv'

run_summary = pd.DataFrame({
    'timestamp': [datetime.now().isoformat()],
    'max_depth': [best_params['max_depth']],
    'min_samples_leaf': [best_params['min_samples_leaf']],
    'rmse': [best_metrics['rmse']],
    'mae': [best_metrics['mae']],
    'r2': [best_metrics['r2']],
    'num_test_samples': [len(test_data_Y)]
})

if csv_path.exists():
    existing = pd.read_csv(csv_path)
    run_summary = pd.concat([existing, run_summary], ignore_index=True)

run_summary.to_csv(csv_path, index=False)
print(f"\nRun summary saved to {csv_path}")
