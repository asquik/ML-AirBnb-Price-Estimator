"""
Train and compare 6 tabular models on Airbnb price prediction:
1. Decision Tree (baseline)
2. LightGBM (sophisticated tree ensemble)
3. Ridge Regression (linear)
4. Polynomial Features + Ridge (quadratic/power-based)
5. MLP (Simple) - shallow neural network
6. MLP (Sophisticated) - deeper neural network with regularization

Uses deterministic train/val/test split (80/10/10) with proper:
- Hyperparameter tuning on validation set (no test leakage)
- Final evaluation on held-out test set
- Box-Cox transformation for price normalization
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
from datetime import datetime
import warnings
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

warnings.filterwarnings('ignore')

# ============================================================================
# FEATURE FLAG FOR DEEP LEARNING
# ============================================================================
ENABLE_DEEP_LEARNING = True  # Set to True to run MLP models (requires PyTorch)

# Load train/val/test data
print("Loading train/val/test tabular data...")
train_df = pd.read_parquet('data/train_tabular.parquet')
val_df = pd.read_parquet('data/val_tabular.parquet')
test_df = pd.read_parquet('data/test_tabular.parquet')

# Load Box-Cox price transformer (fit on train only during preprocessing)
# This stabilizes the extreme variance in raw price ($11-$26,724 → log scale)
price_transformer = joblib.load('data/price_transformer.joblib')

# Feature columns (preprocessed: encoded categoricals, scaled numerics)
feature_cols = ['room_type', 'neighbourhood_cleansed', 'accommodates', 'bathrooms', 'bedrooms', 'minimum_nights', 'season_ordinal']

# Extract X,y for each split
# Using price_bc (Box-Cox transformed) to stabilize variance
# Raw price has std=$396 and outliers up to $26K; Box-Cox normalizes this to std~0.29
X_train = train_df[feature_cols].copy()
y_train = train_df['price_bc'].values  # Use transformed price (stabilized)
y_train_raw = train_df['price'].values  # Keep raw for inverse-transform at end

X_val = val_df[feature_cols].copy()
y_val = val_df['price_bc'].values
y_val_raw = val_df['price'].values

X_test = test_df[feature_cols].copy()
y_test = test_df['price_bc'].values
y_test_raw = test_df['price'].values

print(f"✅ Data loaded: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
print(f"   Features: {feature_cols}")
print(f"   Target: price_bc (Box-Cox transformed for variance stabilization)")
print(f"   Raw price stats: mean=${y_train_raw.mean():.2f}, std=${y_train_raw.std():.2f}, max=${y_train_raw.max():.0f}")
print(f"   Transformed price stats: mean={y_train.mean():.3f}, std={y_train.std():.3f}\n")


# ============================================================================
# PYTORCH SETUP FOR DEEP LEARNING
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

# Feature engineering for deep learning: separate numeric and categorical
NUMERIC_COLS_DL = ['accommodates', 'bathrooms', 'bedrooms', 'minimum_nights', 'season_ordinal']
CATEGORICAL_COLS_DL = ['room_type', 'neighbourhood_cleansed']

# Fit encoders on full train data
numeric_scaler_dl = StandardScaler()
numeric_scaler_dl.fit(train_df[NUMERIC_COLS_DL])

room_type_encoder = LabelEncoder()
room_type_encoder.fit(train_df['room_type'])

neighbourhood_encoder = LabelEncoder()
neighbourhood_encoder.fit(train_df['neighbourhood_cleansed'])


class AirbnbTabularDataset(Dataset):
    """PyTorch Dataset for tabular Airbnb data."""
    
    def __init__(self, df, numeric_scaler, room_type_enc, neighbourhood_enc):
        self.numeric_data = numeric_scaler.transform(df[NUMERIC_COLS_DL]).astype(np.float32)
        self.room_type = room_type_enc.transform(df['room_type']).reshape(-1, 1).astype(np.int64)
        
        # Handle unknown neighbourhoods
        self.neighbourhood = np.zeros((len(df), 1), dtype=np.int64)
        for i, n in enumerate(df['neighbourhood_cleansed'].values):
            try:
                self.neighbourhood[i, 0] = neighbourhood_enc.transform([n])[0]
            except ValueError:
                self.neighbourhood[i, 0] = -1
        
        self.targets = df['price_bc'].values.astype(np.float32)
    
    def __len__(self):
        return len(self.numeric_data)
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.numeric_data[idx]),
            torch.from_numpy(self.room_type[idx]),
            torch.from_numpy(self.neighbourhood[idx]),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )


class TabularMLP(nn.Module):
    """MLP for tabular data with embeddings for categorical features."""
    
    def __init__(self, num_numeric, num_room_types, num_neighbourhoods, 
                 hidden_dims, embedding_dims, dropout_rate):
        super().__init__()
        
        self.room_type_emb = nn.Embedding(num_room_types, embedding_dims[0])
        self.neighbourhood_emb = nn.Embedding(num_neighbourhoods + 1, embedding_dims[1])
        
        total_input = num_numeric + embedding_dims[0] + embedding_dims[1]
        
        layers = []
        input_size = total_input
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_size, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_dim
        
        layers.append(nn.Linear(input_size, 1))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, numeric, room_type, neighbourhood):
        room_emb = self.room_type_emb(room_type).squeeze(1)
        neigh_emb = self.neighbourhood_emb(neighbourhood).squeeze(1)
        x = torch.cat([numeric, room_emb, neigh_emb], dim=1)
        return self.mlp(x)


def train_torch_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=15):
    """Train PyTorch model with early stopping."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    print(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'LR':<12} {'Patience':<10}")
    print("-" * 60)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for numeric, room_type, neighbourhood, target in train_loader:
            numeric, room_type, neighbourhood, target = (
                numeric.to(device), room_type.to(device), neighbourhood.to(device), target.to(device).unsqueeze(1)
            )
            optimizer.zero_grad()
            pred = model(numeric, room_type, neighbourhood)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(numeric)
        
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for numeric, room_type, neighbourhood, target in val_loader:
                numeric, room_type, neighbourhood, target = (
                    numeric.to(device), room_type.to(device), neighbourhood.to(device), target.to(device).unsqueeze(1)
                )
                pred = model(numeric, room_type, neighbourhood)
                loss = criterion(pred, target)
                val_loss += loss.item() * len(numeric)
        
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"{epoch+1:<8} {train_loss:<15.6f} {val_loss:<15.6f} {current_lr:<12.6f} {patience_counter:<10}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"✅ Early stopping at epoch {epoch+1}")
            break
    
    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate_torch_model(model, dataset, y_raw, model_name):
    """Evaluate PyTorch model and return metrics in raw dollars."""
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    model.eval()
    
    predictions_bc = []
    with torch.no_grad():
        for numeric, room_type, neighbourhood, _ in loader:
            numeric, room_type, neighbourhood = numeric.to(device), room_type.to(device), neighbourhood.to(device)
            pred = model(numeric, room_type, neighbourhood)
            predictions_bc.append(pred.cpu().numpy().ravel())
    
    y_pred_bc = np.concatenate(predictions_bc)
    y_pred_raw = price_transformer.inverse_transform(y_pred_bc.reshape(-1, 1)).ravel()
    
    rmse = np.sqrt(mean_squared_error(y_raw, y_pred_raw))
    mae = mean_absolute_error(y_raw, y_pred_raw)
    r2 = r2_score(y_raw, y_pred_raw)
    
    print(f"\n{model_name}")
    print("=" * 80)
    print(f"Test (raw $):        RMSE=${rmse:.2f}  MAE=${mae:.2f}  R²={r2:.4f}")
    
    return {'test_rmse_raw': rmse, 'test_mae_raw': mae, 'test_r2_raw': r2}


def evaluate_model(model, X_val, y_val, y_val_raw, X_test, y_test, y_test_raw, model_name):
    """Evaluate sklearn model on validation and test sets.
    
    Reports metrics in both transformed (Box-Cox) and raw dollar scales.
    Box-Cox metrics help compare how well models fit the normalized distribution.
    Raw dollar metrics are more interpretable for real-world use.
    """
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # === Metrics on Box-Cox transformed scale (what the model actually optimizes) ===
    val_r2_bc = r2_score(y_val, y_val_pred)
    test_r2_bc = r2_score(y_test, y_test_pred)
    
    # === Inverse-transform predictions back to raw dollars for interpretability ===
    y_val_pred_raw = price_transformer.inverse_transform(y_val_pred.reshape(-1, 1)).ravel()
    y_test_pred_raw = price_transformer.inverse_transform(y_test_pred.reshape(-1, 1)).ravel()
    
    val_rmse_raw = np.sqrt(mean_squared_error(y_val_raw, y_val_pred_raw))
    val_mae_raw = mean_absolute_error(y_val_raw, y_val_pred_raw)
    val_r2_raw = r2_score(y_val_raw, y_val_pred_raw)
    
    test_rmse_raw = np.sqrt(mean_squared_error(y_test_raw, y_test_pred_raw))
    test_mae_raw = mean_absolute_error(y_test_raw, y_test_pred_raw)
    test_r2_raw = r2_score(y_test_raw, y_test_pred_raw)
    
    print(f"\n{model_name}")
    print("=" * 80)
    print(f"Validation (raw $):  RMSE=${val_rmse_raw:.2f}  MAE=${val_mae_raw:.2f}  R²={val_r2_raw:.4f}")
    print(f"Test (raw $):        RMSE=${test_rmse_raw:.2f}  MAE=${test_mae_raw:.2f}  R²={test_r2_raw:.4f}")
    print(f"[Box-Cox R²: val={val_r2_bc:.4f}, test={test_r2_bc:.4f}]")
    
    return {
        'model_name': model_name,
        'val_rmse_raw': val_rmse_raw,
        'val_mae_raw': val_mae_raw,
        'val_r2_raw': val_r2_raw,
        'test_rmse_raw': test_rmse_raw,
        'test_mae_raw': test_mae_raw,
        'test_r2_raw': test_r2_raw,
        'test_r2_bc': test_r2_bc,
        'best_params': str(model.get_params()) if hasattr(model, 'get_params') else 'N/A'
    }


all_model_results = []


# ============================================================================
# 1. DECISION TREE BASELINE
# ============================================================================
print("\n" + "=" * 80)
print("1. DECISION TREE BASELINE")
print("=" * 80)

best_dt_model = None
best_dt_params = {}
best_dt_val_rmse = float('inf')

max_depths = [3, 5, 8, 12, 15, 20, 25, 30]
min_samples_leafs = [2, 5, 10, 20, 30]

print(f"\nHyperparameter sweep: max_depth={max_depths}, min_samples_leaf={min_samples_leafs}")
print(f"{'max_depth':<12} {'min_samples_leaf':<18} {'Val RMSE':<12} {'Val MAE':<12} {'Val R²':<10}")
print("-" * 70)

for max_depth in max_depths:
    for min_samples_leaf in min_samples_leafs:
        dt = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        dt.fit(X_train, y_train)
        y_val_pred = dt.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        print(f"{max_depth:<12} {min_samples_leaf:<18} {val_rmse:<12.4f} {val_mae:<12.4f} {val_r2:<10.4f}")
        
        if val_rmse < best_dt_val_rmse:
            best_dt_val_rmse = val_rmse
            best_dt_model = dt
            best_dt_params = {'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf}

print("-" * 70)
print(f"✅ Best Decision Tree: max_depth={best_dt_params['max_depth']}, min_samples_leaf={best_dt_params['min_samples_leaf']}")

dt_result = evaluate_model(best_dt_model, X_val, y_val, y_val_raw, X_test, y_test, y_test_raw,
                            f"Decision Tree (max_depth={best_dt_params['max_depth']}, min_samples_leaf={best_dt_params['min_samples_leaf']})")
all_model_results.append(dt_result)


# ============================================================================
# 2. GRADIENT BOOSTING (sklearn or LightGBM)
# ============================================================================
if HAS_LIGHTGBM:
    print("\n" + "=" * 80)
    print("2. LIGHTGBM (GRADIENT BOOSTING)")
    print("=" * 80)
    
    best_lgb_model = None
    best_lgb_params = {}
    best_lgb_val_rmse = float('inf')
    
    num_leaves_list = [15, 31, 50, 100]
    learning_rates = [0.01, 0.05, 0.1]
    n_estimators_list = [50, 100, 200, 300]
    
    print(f"\nHyperparameter sweep: num_leaves={num_leaves_list}, learning_rate={learning_rates}, n_estimators={n_estimators_list}")
    print(f"{'num_leaves':<12} {'learning_rate':<16} {'n_estimators':<14} {'Val RMSE':<12} {'Val R²':<10}")
    print("-" * 70)
    
    for num_leaves in num_leaves_list:
        for learning_rate in learning_rates:
            for n_estimators in [100, 200]:  # Reduced to speed up search
                lgb_model = lgb.LGBMRegressor(
                    num_leaves=num_leaves,
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    random_state=42,
                    verbose=-1,
                    force_col_wise=True
                )
                lgb_model.fit(X_train, y_train)
                y_val_pred = lgb_model.predict(X_val)
                val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                val_r2 = r2_score(y_val, y_val_pred)
                
                print(f"{num_leaves:<12} {learning_rate:<16.3f} {n_estimators:<14} {val_rmse:<12.4f} {val_r2:<10.4f}")
                
                if val_rmse < best_lgb_val_rmse:
                    best_lgb_val_rmse = val_rmse
                    best_lgb_model = lgb_model
                    best_lgb_params = {
                        'num_leaves': num_leaves,
                        'learning_rate': learning_rate,
                        'n_estimators': n_estimators
                    }
    
    print("-" * 70)
    print(f"✅ Best LightGBM: num_leaves={best_lgb_params['num_leaves']}, learning_rate={best_lgb_params['learning_rate']}, n_estimators={best_lgb_params['n_estimators']}")
    
    lgb_result = evaluate_model(best_lgb_model, X_val, y_val, y_val_raw, X_test, y_test, y_test_raw,
                                f"LightGBM (num_leaves={best_lgb_params['num_leaves']}, lr={best_lgb_params['learning_rate']}, n_est={best_lgb_params['n_estimators']})")
    all_model_results.append(lgb_result)

else:
    print("\n" + "=" * 80)
    print("2. GRADIENT BOOSTING (SKLEARN - scikit-learn GradientBoostingRegressor)")
    print("=" * 80)
    
    best_gb_model = None
    best_gb_params = {}
    best_gb_val_rmse = float('inf')
    
    max_depths = [3, 5, 7]
    learning_rates = [0.01, 0.05, 0.1]
    n_estimators_list = [50, 100, 200]
    
    print(f"\nHyperparameter sweep: max_depth={max_depths}, learning_rate={learning_rates}, n_estimators={n_estimators_list}")
    print(f"{'max_depth':<12} {'learning_rate':<16} {'n_estimators':<14} {'Val RMSE':<12} {'Val R²':<10}")
    print("-" * 70)
    
    for max_depth in max_depths:
        for learning_rate in learning_rates:
            for n_estimators in [100, 200]:  # Reduced to speed up search
                gb_model = GradientBoostingRegressor(
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    random_state=42
                )
                gb_model.fit(X_train, y_train)
                y_val_pred = gb_model.predict(X_val)
                val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                val_r2 = r2_score(y_val, y_val_pred)
                
                print(f"{max_depth:<12} {learning_rate:<16.3f} {n_estimators:<14} {val_rmse:<12.4f} {val_r2:<10.4f}")
                
                if val_rmse < best_gb_val_rmse:
                    best_gb_val_rmse = val_rmse
                    best_gb_model = gb_model
                    best_gb_params = {
                        'max_depth': max_depth,
                        'learning_rate': learning_rate,
                        'n_estimators': n_estimators
                    }
    
    print("-" * 70)
    print(f"✅ Best GradientBoosting: max_depth={best_gb_params['max_depth']}, learning_rate={best_gb_params['learning_rate']}, n_estimators={best_gb_params['n_estimators']}")
    
    gb_result = evaluate_model(best_gb_model, X_val, y_val, y_val_raw, X_test, y_test, y_test_raw,
                               f"GradientBoosting (max_depth={best_gb_params['max_depth']}, lr={best_gb_params['learning_rate']}, n_est={best_gb_params['n_estimators']})")
    all_model_results.append(gb_result)


# ============================================================================
# 3. RIDGE REGRESSION (LINEAR)
# ============================================================================
print("\n" + "=" * 80)
print("3. RIDGE REGRESSION (LINEAR REGRESSION WITH L2 REGULARIZATION)")
print("=" * 80)

best_ridge_model = None
best_ridge_alpha = None
best_ridge_val_rmse = float('inf')

alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

print(f"\nHyperparameter sweep: alpha={alphas}")
print(f"{'alpha':<12} {'Val RMSE':<12} {'Val MAE':<12} {'Val R²':<10}")
print("-" * 70)

for alpha in alphas:
    ridge = Ridge(alpha=alpha, random_state=42)
    ridge.fit(X_train, y_train)
    y_val_pred = ridge.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"{alpha:<12.4f} {val_rmse:<12.4f} {val_mae:<12.4f} {val_r2:<10.4f}")
    
    if val_rmse < best_ridge_val_rmse:
        best_ridge_val_rmse = val_rmse
        best_ridge_model = ridge
        best_ridge_alpha = alpha

print("-" * 70)
print(f"✅ Best Ridge: alpha={best_ridge_alpha}")

ridge_result = evaluate_model(best_ridge_model, X_val, y_val, y_val_raw, X_test, y_test, y_test_raw,
                              f"Ridge Regression (alpha={best_ridge_alpha})")
all_model_results.append(ridge_result)


# ============================================================================
# 4. POLYNOMIAL FEATURES + RIDGE (QUADRATIC)
# ============================================================================
print("\n" + "=" * 80)
print("4. POLYNOMIAL FEATURES (DEGREE 2) + RIDGE (QUADRATIC/POWER-BASED)")
print("=" * 80)

best_poly_model = None
best_poly_params = {}
best_poly_val_rmse = float('inf')

poly_degrees = [2, 3]
poly_alphas = [0.001, 0.01, 0.1, 1.0, 10.0]

print(f"\nHyperparameter sweep: degree={poly_degrees}, alpha={poly_alphas}")
print(f"{'degree':<12} {'alpha':<12} {'Val RMSE':<12} {'Val MAE':<12} {'Val R²':<10}")
print("-" * 70)

for degree in poly_degrees:
    for alpha in poly_alphas:
        # Pipeline: PolynomialFeatures → Ridge
        poly_model = Pipeline([
            ('poly_features', PolynomialFeatures(degree=degree, include_bias=False)),
            ('ridge', Ridge(alpha=alpha, random_state=42))
        ])
        poly_model.fit(X_train, y_train)
        y_val_pred = poly_model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        print(f"{degree:<12} {alpha:<12.4f} {val_rmse:<12.4f} {val_mae:<12.4f} {val_r2:<10.4f}")
        
        if val_rmse < best_poly_val_rmse:
            best_poly_val_rmse = val_rmse
            best_poly_model = poly_model
            best_poly_params = {'degree': degree, 'alpha': alpha}

print("-" * 70)
print(f"✅ Best Polynomial+Ridge: degree={best_poly_params['degree']}, alpha={best_poly_params['alpha']}")

poly_result = evaluate_model(best_poly_model, X_val, y_val, y_val_raw, X_test, y_test, y_test_raw,
                             f"Polynomial (degree={best_poly_params['degree']}) + Ridge (alpha={best_poly_params['alpha']})")
all_model_results.append(poly_result)


# ============================================================================
# 5. DEEP LEARNING - MLP (SIMPLE)
# ============================================================================
if ENABLE_DEEP_LEARNING:
    print("\n" + "=" * 80)
    print("5. DEEP LEARNING - MLP (SIMPLE)")
    print("=" * 80)
    print("Architecture: 2 hidden layers [64, 32], small embeddings [4, 8]")

    train_dlset = AirbnbTabularDataset(train_df, numeric_scaler_dl, room_type_encoder, neighbourhood_encoder)
    val_dlset = AirbnbTabularDataset(val_df, numeric_scaler_dl, room_type_encoder, neighbourhood_encoder)
    test_dlset = AirbnbTabularDataset(test_df, numeric_scaler_dl, room_type_encoder, neighbourhood_encoder)

    train_loader = DataLoader(train_dlset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dlset, batch_size=256, shuffle=False)

    mlp_simple = TabularMLP(
        num_numeric=len(NUMERIC_COLS_DL),
        num_room_types=len(room_type_encoder.classes_),
        num_neighbourhoods=len(neighbourhood_encoder.classes_),
        hidden_dims=[64, 32],
        embedding_dims=[4, 8],
        dropout_rate=0.2
    ).to(device)

    mlp_simple = train_torch_model(mlp_simple, train_loader, val_loader, epochs=150, lr=0.005, patience=20)
    mlp_simple_result = evaluate_torch_model(mlp_simple, test_dlset, y_test_raw, "MLP (Simple)")
    mlp_simple_result['model_name'] = "MLP (Simple) [64,32] embeddings=[4,8]"
    mlp_simple_result['best_params'] = "layers=[64,32], embeddings=[4,8], lr=0.005, dropout=0.2"
    all_model_results.append(mlp_simple_result)


# ============================================================================
# 6. DEEP LEARNING - MLP (SOPHISTICATED)
# ============================================================================
if ENABLE_DEEP_LEARNING:
    print("\n" + "=" * 80)
    print("6. DEEP LEARNING - MLP (SOPHISTICATED)")
    print("=" * 80)
    print("Architecture: 4 hidden layers [256, 128, 64, 32], larger embeddings [8, 16]")

    mlp_sophisticated = TabularMLP(
        num_numeric=len(NUMERIC_COLS_DL),
        num_room_types=len(room_type_encoder.classes_),
        num_neighbourhoods=len(neighbourhood_encoder.classes_),
        hidden_dims=[256, 128, 64, 32],
        embedding_dims=[8, 16],
        dropout_rate=0.4
    ).to(device)

    mlp_sophisticated = train_torch_model(mlp_sophisticated, train_loader, val_loader, epochs=150, lr=0.001, patience=20)
    mlp_sophisticated_result = evaluate_torch_model(mlp_sophisticated, test_dlset, y_test_raw, "MLP (Sophisticated)")
    mlp_sophisticated_result['model_name'] = "MLP (Sophisticated) [256,128,64,32] embeddings=[8,16]"
    mlp_sophisticated_result['best_params'] = "layers=[256,128,64,32], embeddings=[8,16], lr=0.001, dropout=0.4"
    all_model_results.append(mlp_sophisticated_result)


# ============================================================================
# SUMMARY AND COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY: ALL MODELS COMPARISON ON TEST SET (RAW DOLLARS)")
print("=" * 80)

summary_df = pd.DataFrame(all_model_results)
print(summary_df[['model_name', 'test_rmse_raw', 'test_mae_raw', 'test_r2_raw']].to_string(index=False))

print("\n" + "=" * 80)
print("BEST MODEL BY TEST R² (on raw dollars):")
best_idx = summary_df['test_r2_raw'].idxmax()
best_model_name = summary_df.loc[best_idx, 'model_name']
best_test_rmse = summary_df.loc[best_idx, 'test_rmse_raw']
best_test_r2 = summary_df.loc[best_idx, 'test_r2_raw']
print(f"✅ {best_model_name}")
print(f"   Test RMSE: ${best_test_rmse:.2f}")
print(f"   Test R²: {best_test_r2:.4f}")
print("=" * 80)

# Save results to CSV
output_dir = Path('outputs')
output_dir.mkdir(exist_ok=True)
csv_path = output_dir / 'model_runs.csv'

# Add metadata
summary_df['timestamp'] = datetime.now().isoformat()
summary_df['train_size'] = len(X_train)
summary_df['val_size'] = len(X_val)
summary_df['test_size'] = len(X_test)

if csv_path.exists():
    existing = pd.read_csv(csv_path)
    summary_df = pd.concat([existing, summary_df], ignore_index=True)

summary_df.to_csv(csv_path, index=False)
print(f"\n✅ Results saved to {csv_path}")

# Save best model summary
print(f"\n📊 All results (in raw dollars):")
print(summary_df[['timestamp', 'model_name', 'test_rmse_raw', 'test_mae_raw', 'test_r2_raw']].to_string(index=False))
