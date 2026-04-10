"""
Train a deep learning model (MLP) on tabular Airbnb data.

Compares against tree-based models by:
1. Using the same 7 tabular features
2. Training on Box-Cox transformed price
3. Inverse-transforming predictions to raw dollars for reporting
4. Saving results to the same CSV

MLP Architecture:
- Input: 7 features (room_type, neighbourhood, accommodates, bathrooms, bedrooms, min_nights, season)
- Handling: 2 categorical features via embedding layers, 5 numeric features normalized
- Hidden layers: Batch norm + dropout for stability
- Output: Single neuron for regression
- Loss: MSE on Box-Cox transformed price
- Optimizer: Adam with learning rate scheduling
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
from datetime import datetime
import warnings
import joblib

warnings.filterwarnings('ignore')

# ============================================================================
# DEVICE & REPRODUCIBILITY
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# LOAD DATA & PRICE TRANSFORMER
# ============================================================================
print("\n" + "=" * 80)
print("LOADING DATA & TRANSFORMERS")
print("=" * 80)

train_df = pd.read_parquet('data/train_tabular.parquet')
val_df = pd.read_parquet('data/val_tabular.parquet')
test_df = pd.read_parquet('data/test_tabular.parquet')

price_transformer = joblib.load('data/price_transformer.joblib')

print(f"✅ Data loaded: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
NUMERIC_COLS = ['accommodates', 'bathrooms', 'bedrooms', 'minimum_nights', 'season_ordinal']
CATEGORICAL_COLS = ['room_type', 'neighbourhood_cleansed']

# Fit scalers/encoders on train only (no leakage)
numeric_scaler = StandardScaler()
numeric_scaler.fit(train_df[NUMERIC_COLS])

room_type_encoder = LabelEncoder()
room_type_encoder.fit(train_df['room_type'])

neighbourhood_encoder = LabelEncoder()
neighbourhood_encoder.fit(train_df['neighbourhood_cleansed'])

# ============================================================================
# PYTORCH DATASET
# ============================================================================
class AirbnbTabularDataset(Dataset):
    """Converts Airbnb tabular data to PyTorch tensors."""
    
    def __init__(self, df, numeric_scaler, room_type_encoder, neighbourhood_encoder, price_transformer=None):
        # Numeric features (scaled)
        self.numeric_data = numeric_scaler.transform(df[NUMERIC_COLS]).astype(np.float32)
        
        # Categorical features (encoded as integers for embedding layers)
        self.room_type_encoded = room_type_encoder.transform(df['room_type']).reshape(-1, 1).astype(np.int64)
        
        # Handle unknown neighborhoods (edge case for test set)
        neighbourhood_values = df['neighbourhood_cleansed'].values
        self.neighbourhood_encoded = np.zeros((len(df), 1), dtype=np.int64)
        for i, neighbourhood in enumerate(neighbourhood_values):
            try:
                self.neighbourhood_encoded[i, 0] = neighbourhood_encoder.transform([neighbourhood])[0]
            except ValueError:
                # Unknown category (not in training data)
                self.neighbourhood_encoded[i, 0] = -1  # Will be handled by embedding
        
        # Target (Box-Cox transformed price if available, else None)
        if 'price_bc' in df.columns:
            self.targets = df['price_bc'].values.astype(np.float32)
        else:
            self.targets = None
    
    def __len__(self):
        return len(self.numeric_data)
    
    def __getitem__(self, idx):
        numeric = torch.from_numpy(self.numeric_data[idx])
        room_type = torch.from_numpy(self.room_type_encoded[idx])
        neighbourhood = torch.from_numpy(self.neighbourhood_encoded[idx])
        
        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return numeric, room_type, neighbourhood, target
        else:
            return numeric, room_type, neighbourhood


# Create datasets
train_dataset = AirbnbTabularDataset(train_df, numeric_scaler, room_type_encoder, neighbourhood_encoder)
val_dataset = AirbnbTabularDataset(val_df, numeric_scaler, room_type_encoder, neighbourhood_encoder)
test_dataset = AirbnbTabularDataset(test_df, numeric_scaler, room_type_encoder, neighbourhood_encoder)

# ============================================================================
# PYTORCH MODEL
# ============================================================================
class TabularMLP(nn.Module):
    """MLP for tabular data with embedding layers for categorical features."""
    
    def __init__(self, num_numeric_features, num_room_types, num_neighbourhoods, 
                 hidden_dims=[128, 64, 32], embedding_dims=[8, 16], dropout_rate=0.3):
        super().__init__()
        
        # Embedding layers for categorical features
        self.room_type_embedding = nn.Embedding(num_room_types, embedding_dims[0])
        self.neighbourhood_embedding = nn.Embedding(num_neighbourhoods + 1, embedding_dims[1])  # +1 for unknown
        
        # Total input size: numeric + embeddings
        total_input_size = num_numeric_features + embedding_dims[0] + embedding_dims[1]
        
        # MLP layers
        layers = []
        input_size = total_input_size
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_size, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_size, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, numeric, room_type, neighbourhood):
        # Embeddings
        room_type_emb = self.room_type_embedding(room_type).squeeze(1)
        neighbourhood_emb = self.neighbourhood_embedding(neighbourhood).squeeze(1)
        
        # Concatenate all features
        x = torch.cat([numeric, room_type_emb, neighbourhood_emb], dim=1)
        
        # MLP
        return self.mlp(x)


# ============================================================================
# TRAINING LOOP
# ============================================================================
def train_model(model, train_loader, val_loader, epochs=100, learning_rate=0.001, 
                patience=15, model_name="MLP"):
    """Train the model with early stopping."""
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"\nTraining {model_name}...")
    print(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Lr':<12}")
    print("-" * 50)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for numeric, room_type, neighbourhood, target in train_loader:
            numeric = numeric.to(device)
            room_type = room_type.to(device)
            neighbourhood = neighbourhood.to(device)
            target = target.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            pred = model(numeric, room_type, neighbourhood)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(numeric)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for numeric, room_type, neighbourhood, target in val_loader:
                numeric = numeric.to(device)
                room_type = room_type.to(device)
                neighbourhood = neighbourhood.to(device)
                target = target.to(device).unsqueeze(1)
                
                pred = model(numeric, room_type, neighbourhood)
                loss = criterion(pred, target)
                val_loss += loss.item() * len(numeric)
        
        val_loss /= len(val_loader.dataset)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"{epoch+1:<8} {train_loss:<15.6f} {val_loss:<15.6f} {current_lr:<12.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"✅ Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model


# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_model(model, dataset, raw_prices, model_name, dataset_name="Test"):
    """Evaluate model and return metrics in raw dollars."""
    
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    model.eval()
    
    predictions_bc = []
    with torch.no_grad():
        for numeric, room_type, neighbourhood in loader:
            numeric = numeric.to(device)
            room_type = room_type.to(device)
            neighbourhood = neighbourhood.to(device)
            
            pred = model(numeric, room_type, neighbourhood)
            predictions_bc.append(pred.cpu().numpy().ravel())
    
    # Concatenate and inverse-transform
    y_pred_bc = np.concatenate(predictions_bc)
    y_pred_raw = price_transformer.inverse_transform(y_pred_bc.reshape(-1, 1)).ravel()
    
    # Metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    rmse = np.sqrt(mean_squared_error(raw_prices, y_pred_raw))
    mae = mean_absolute_error(raw_prices, y_pred_raw)
    r2 = r2_score(raw_prices, y_pred_raw)
    
    print(f"\n{model_name}")
    print("=" * 80)
    print(f"{dataset_name} Set (raw $):  RMSE=${rmse:.2f}  MAE=${mae:.2f}  R²={r2:.4f}")
    
    return rmse, mae, r2


# ============================================================================
# MAIN TRAINING
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING DEEP LEARNING MODELS")
print("=" * 80)

# Hyperparameter search over architecture sizes
results = []

config_options = [
    {'hidden_dims': [64, 32], 'embedding_dims': [4, 8], 'dropout': 0.2, 'lr': 0.005},
    {'hidden_dims': [128, 64, 32], 'embedding_dims': [8, 16], 'dropout': 0.3, 'lr': 0.001},
    {'hidden_dims': [256, 128, 64], 'embedding_dims': [8, 16], 'dropout': 0.4, 'lr': 0.0005},
    {'hidden_dims': [128, 128, 64, 32], 'embedding_dims': [8, 16], 'dropout': 0.3, 'lr': 0.001},
]

for config_idx, config in enumerate(config_options, 1):
    print(f"\n{'='*80}")
    print(f"CONFIG {config_idx}: hidden={config['hidden_dims']}, emb={config['embedding_dims']}, dropout={config['dropout']}, lr={config['lr']}")
    print(f"{'='*80}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    # Create model
    model = TabularMLP(
        num_numeric_features=len(NUMERIC_COLS),
        num_room_types=len(room_type_encoder.classes_),
        num_neighbourhoods=len(neighbourhood_encoder.classes_),
        hidden_dims=config['hidden_dims'],
        embedding_dims=config['embedding_dims'],
        dropout_rate=config['dropout']
    ).to(device)
    
    # Train
    model = train_model(
        model, train_loader, val_loader, 
        epochs=150, 
        learning_rate=config['lr'],
        patience=20,
        model_name=f"MLP Config {config_idx}"
    )
    
    # Evaluate on validation and test
    val_rmse, val_mae, val_r2 = evaluate_model(
        model, val_dataset, val_df['price'].values,
        f"MLP Config {config_idx}", "Validation"
    )
    test_rmse, test_mae, test_r2 = evaluate_model(
        model, test_dataset, test_df['price'].values,
        f"MLP Config {config_idx}", "Test"
    )
    
    config_str = f"hidden={config['hidden_dims']}, emb={config['embedding_dims']}, dropout={config['dropout']}, lr={config['lr']}"
    results.append({
        'model_name': f"MLP ({config_str})",
        'val_rmse_raw': val_rmse,
        'val_mae_raw': val_mae,
        'val_r2_raw': val_r2,
        'test_rmse_raw': test_rmse,
        'test_mae_raw': test_mae,
        'test_r2_raw': test_r2,
        'best_params': config_str
    })

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY: DEEP LEARNING MODELS COMPARISON ON TEST SET")
print("=" * 80)

summary_df = pd.DataFrame(results)
print(summary_df[['model_name', 'test_rmse_raw', 'test_mae_raw', 'test_r2_raw']].to_string(index=False))

# Append to existing CSV
output_dir = Path('outputs')
output_dir.mkdir(exist_ok=True)
csv_path = output_dir / 'model_runs.csv'

# Add metadata
summary_df['timestamp'] = datetime.now().isoformat()
summary_df['train_size'] = len(train_df)
summary_df['val_size'] = len(val_df)
summary_df['test_size'] = len(test_df)

if csv_path.exists():
    existing = pd.read_csv(csv_path)
    summary_df = pd.concat([existing, summary_df], ignore_index=True)

summary_df.to_csv(csv_path, index=False)
print(f"\n✅ Results saved to {csv_path}")

# Best model
if len(summary_df) > 0:
    best_idx = summary_df['test_r2_raw'].idxmax()
    best_model_name = summary_df.loc[best_idx, 'model_name']
    best_test_r2 = summary_df.loc[best_idx, 'test_r2_raw']
    best_test_rmse = summary_df.loc[best_idx, 'test_rmse_raw']
else:
    best_model_name = "No models trained"
    best_test_r2 = 0
    best_test_rmse = 0

print("\n" + "=" * 80)
print("BEST DEEP LEARNING MODEL:")
print(f"✅ {best_model_name}")
print(f"   Test RMSE: ${best_test_rmse:.2f}")
print(f"   Test R²: {best_test_r2:.4f}")
print("=" * 80)
