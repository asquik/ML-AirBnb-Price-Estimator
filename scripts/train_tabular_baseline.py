"""Training template with step-by-step comments (no implementation code).

Purpose: guide your implementation for the tabular decision tree baseline.

You should edit this file and replace comments with real Python code as you learn.
"""

import pandas as pd
train_df = pd.read_parquet('data/train.parquet')
print(train_df.shape)
print(train_df.head())

# 2) PROCESS DATA
def preprocess_tabular(df, encoders=None, fit=False):
    train_data_X = df[['room_type','neighbourhood_cleansed','accommodates','bathrooms','bedrooms','minimum_nights']]
    train_data_Y = df[['price']]
# - Handle missing values (e.g., fillna with median for numeric columns)
# - Encode categorical columns:
#     Use sklearn.preprocessing.LabelEncoder or pd.get_dummies.
# - If fit=True, fit encoders on training data and return them along with X,y
# - If fit=False, use provided encoders to transform test data
# - Convert final features to numeric numpy arrays (X) and target to y

# 3) TRAIN MODEL
# - Import DecisionTreeRegressor from sklearn.tree
# - Define train_and_evaluate(...) function:
#      Inputs: X_train, y_train, X_test, y_test
#      Hyperparameter grid: max_depth=[8,12,15], min_samples_leaf=[10,20]
#      Loop grid:
#        model = DecisionTreeRegressor(max_depth=..., min_samples_leaf=..., random_state=42)
#        model.fit(X_train,y_train)
#        y_pred = model.predict(X_test)
#        compute rmse, mae, r2 with sklearn.metrics
#      Track best model by lowest rmse
#      Return best model and metrics dict

# 4) VERIFY MODEL
# - Print best hyperparameters and final test metrics:
#      RMSE, MAE, R2
# - Optionally show a few predictions vs actuals
#     for i in range(5): print(y_test[i], y_pred[i])
# - Save run summary to outputs/model_runs.csv:
#     Create CSV if missing, append new row if exists.
#   Use pandas: pd.read_csv, pd.to_csv
# - Optionally compute feature importances:
#     best_model.feature_importances_

# 5) MAIN EXECUTION BLOCK
# - if __name__ == '__main__':
#     data_dir = Path(...)/'data'
#     outputs_path = Path(...)/'outputs/model_runs.csv'
#     train_df, test_df = load_data(data_dir)
#     X_train, y_train, encoders = preprocess_tabular(train_df, fit=True)
#     X_test, y_test = preprocess_tabular(test_df, encoders=encoders, fit=False)
#     best_model, best_metrics = train_and_evaluate(X_train,y_train,X_test,y_test)
#     log_results(outputs_path, best_metrics)
#     print('done')

# Optional learning notes:
# - In scikit-learn, DecisionTreeRegressor splits on features to minimize squared error in leaves.
# - R² is explained variance: 1 is perfect, 0 is as good as mean prediction.
# - Label encoding for trees is acceptable, but beware of ordering semantics in some models.
# - Keep code modular: separate data loading/preprocessing/training/metrics.
