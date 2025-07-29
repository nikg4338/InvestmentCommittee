# neural_network.py
"""
Modular PyTorch neural network models for the Investment Committee ensemble.
- MLP: For tabular/technical features (RSI, momentum, etc.)
- LSTMNet: For sequential/time series features (past 20 days OHLCV, etc.)
Designed for classification: [Bullish, Neutral, Bearish] or binary [Up/Down].
"""

import os
import torch
import torch.nn as nn
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

def find_best_threshold(y_true, y_probs, metric='f1'):
    """
    Find the best threshold for binary classification to maximize the given metric.
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities 
        metric: Metric to optimize ('f1', 'recall', 'precision')
    
    Returns:
        tuple: (best_threshold, best_score)
    """
    import numpy as np
    from sklearn.metrics import f1_score, recall_score, precision_score
    
    best_t, best_score = 0.5, -1
    for t in np.linspace(0.05, 0.95, 19):
        preds = (y_probs >= t).astype(int)
        try:
            if metric == 'f1':
                score = f1_score(y_true, preds, pos_label=1)
            elif metric == 'recall':
                score = recall_score(y_true, preds, pos_label=1)
            else:
                score = precision_score(y_true, preds, pos_label=1)
            
            if score > best_score:
                best_score, best_t = score, t
        except:
            continue
    
    return best_t, best_score

class FocalLoss(nn.Module):
    """
    Binary focal loss.
    gamma > 1 focuses on hard positives.
    alpha balances classes (similar to pos_weight).
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        ce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_term = (1 - p_t) ** self.gamma
        loss = self.alpha * focal_term * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class BaseNeuralNetwork(nn.Module):
    """
    Base neural net class for all committee models.
    Adds easy save/load and weight initialization.
    """
    def __init__(self):
        super().__init__()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def save(self, path: str) -> bool:
        try:
            torch.save(self.state_dict(), path)
            logger.info(f"Model weights saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def load(self, path: str) -> bool:
        try:
            if not os.path.exists(path):
                logger.error(f"Model path does not exist: {path}")
                return False
            self.load_state_dict(torch.load(path))
            logger.info(f"Model weights loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

class MLP(BaseNeuralNetwork):
    """
    Multi-Layer Perceptron for classification.
    Used for technical/tabular features.
    """
    def __init__(
        self, 
        input_dim: int = 12, 
        output_dim: int = 1,   # 1 for binary, 3 for multi-class
        hidden_dims: Optional[List[int]] = None, 
        dropout: float = 0.2
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.LayerNorm(hdim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, output_dim))
        if output_dim == 1:
            layers.append(nn.Identity())  # Binary classification
        else:
            layers.append(nn.Softmax(dim=1))  # Multi-class classification

        self.network = nn.Sequential(*layers)
        self.initialize_weights()

    def forward(self, x):
        return self.network(x)

class LSTMNet(BaseNeuralNetwork):
    """
    LSTM-based classifier for sequence (time series) input.
    Used for OHLCV data over time.
    """
    def __init__(
        self, 
        input_dim: int = 5, 
        output_dim: int = 3, 
        hidden_dim: int = 64, 
        num_layers: int = 2, 
        dropout: float = 0.2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.softmax = nn.Softmax(dim=1)
        self.initialize_weights()

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]  # Last time step
        out = torch.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

def optuna_search_mlp(X, y, input_size, n_trials=20, max_epochs=40, pretrained_model_path=None, override_lr=None, max_epochs_override=None, params_save_path=None, hidden1_choices=None, hidden2_choices=None):
    """
    Runs Optuna hyperparameter search for an MLP (binary classification).
    Returns: trained PyTorch model, fitted scaler, and best_params dict.
    """
    import optuna
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Log original data distribution
    logger.info(f"Original data shape: X={X.shape}, y={y.shape}")
    logger.info(f"Original label distribution: {np.bincount(y.astype(int))}")
    
    # Variance analysis on features
    feature_variances = np.var(X, axis=0)
    low_variance_features = np.sum(feature_variances < 0.01)
    total_features = len(feature_variances)
    low_variance_ratio = low_variance_features / total_features
    
    logger.info(f"Feature variance analysis:")
    logger.info(f"  - Total features: {total_features}")
    logger.info(f"  - Features with variance < 0.01: {low_variance_features}")
    logger.info(f"  - Low variance ratio: {low_variance_ratio:.3f}")
    
    if low_variance_ratio > 0.8:
        logger.warning(f"⚠️ {low_variance_ratio:.1%} of features have very low variance (< 0.01)")
        logger.warning("This may indicate poor feature quality or scaling issues")
    
    # Stratified train/test split with shuffle=True
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42, shuffle=True
    )
    
    # Log train/val distribution
    logger.info(f"Train set: {X_train.shape}, labels: {np.bincount(y_train.astype(int))}")
    logger.info(f"Val set: {X_val.shape}, labels: {np.bincount(y_val.astype(int))}")
    # Apply SMOTE to the training set to handle class imbalance by resampling
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    logger.info(f"Applied SMOTE. New train set: {X_train_resampled.shape}, labels: {np.bincount(y_train_resampled.astype(int))}")

    # Fit scaler ONLY on the original training data to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled) # Use resampled data
    X_val_scaled = scaler.transform(X_val)

    # Fit scaler ONLY on training data to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    def objective(trial):
        from sklearn.metrics import f1_score
        
        # Set random seed for each trial
        torch.manual_seed(42)
        
        # Expanded search space to find a more powerful architecture
        hidden1 = trial.suggest_categorical("hidden1", [64, 128, 256])
        hidden2 = trial.suggest_categorical("hidden2", [32, 64, 128])
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64])

        # Create DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        class TrialMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Sequential(
                    nn.Linear(input_size, hidden1),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden1, hidden2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden2, 1)
                )

            def forward(self, x):
                return self.model(x).view(-1)

        model = TrialMLP().to(device)
        # Safe pretrained loading
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            try:
                model.load_state_dict(torch.load(pretrained_model_path))
                logger.info(f"✅ Loaded pretrained weights from {pretrained_model_path}")
            except RuntimeError as e:
                logger.warning(f"⚠️ Could not load pretrained weights due to architecture mismatch: {e}")
                logger.warning("Proceeding with randomly initialized weights.")
        
        # Use AdamW optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Use a standard, un-weighted loss function because SMOTE has already balanced the data
        criterion = nn.BCEWithLogitsLoss()
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Training loop with early stopping
        best_val_f1 = -1
        patience = 10
        epochs_no_improve = 0
        
        for epoch in range(max_epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                out = model(batch_X)
                assert out.shape == batch_y.shape, f"Shape mismatch: out={out.shape}, y={batch_y.shape}"
                out = out.view(-1)
                batch_y = batch_y.view(-1)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

            # Validation
            # Validation
            model.eval()
            all_probs = []
            all_labels = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    out = model(batch_X)
                    probs = torch.sigmoid(out.view(-1))
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(batch_y.view(-1).cpu().numpy())

            # Find the best threshold for this epoch's validation probabilities
            # This gives a true measure of the model's separation power
            best_t, val_f1 = find_best_threshold(np.array(all_labels), np.array(all_probs), metric='f1')

            # Handle case where F1 score could not be computed
            if val_f1 < 0:
                val_f1 = 0.0
            try:
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve >= patience:
                    break
            except:
                val_f1 = 0.0
        
        return float(best_val_f1)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print("Best MLP params:", study.best_params)
    best_params = study.best_params

    # === Retrain on ALL data using best params ===
    # Use the new parameter names from the expanded Optuna search
    hidden1 = best_params['hidden1']
    hidden2 = best_params['hidden2']
    dropout = best_params['dropout']
    lr = best_params['lr']
    batch_size = best_params['batch_size']
    
    # Fit scaler on ALL X (but only use training data for fitting)
    full_scaler = StandardScaler()
    X_full_scaled = full_scaler.fit_transform(X)
    X_full_tensor = torch.tensor(X_full_scaled, dtype=torch.float32).to(device)
    y_full_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    full_dataset = TensorDataset(X_full_tensor, y_full_tensor)

    # Split out a validation set for curves (so loss/acc curves are meaningful)
    from sklearn.model_selection import train_test_split
    
    X_train_full, X_val_full, y_train_full, y_val_full = train_test_split(
        X_full_scaled, y, test_size=0.2, stratify=y, random_state=42, shuffle=True
    )
    X_train_tensor = torch.tensor(X_train_full, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_full, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val_full, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val_full, dtype=torch.float32).to(device)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    class FinalMLP(nn.Module):
        def __init__(self):
            super().__init__()
            # Use the best parameters found by Optuna
            hidden1 = best_params['hidden1']
            hidden2 = best_params['hidden2']
            dropout = best_params['dropout']
            
            self.model = nn.Sequential(
                nn.Linear(input_size, hidden1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden2, 1)
            )
        def forward(self, x):
            return self.model(x).view(-1)
        # Add this new method to the class
        def extract_features(self, x):
            """Passes data through the feature extraction part of the network."""
            with torch.no_grad():
                return self.feature_extractor(x).cpu().numpy()

    final_model = FinalMLP().to(device)
     # Initialize final layer bias to reflect class imbalance
    # This helps prevent the model from defaulting to one class early in training
    with torch.no_grad():
        # Use the original, unscaled training data for the prior
        y_train_prior = y_train
        if len(y_train_prior) > 0:
            pos_rate = np.mean(y_train_prior)
            # Avoid log(0)
            if pos_rate > 0 and pos_rate < 1:
                initial_bias = np.log(pos_rate / (1 - pos_rate))
                # The final layer is the last item in the sequential model
                final_model.model[-1].bias.fill_(initial_bias)
                logger.info(f"Set initial bias for final layer to {initial_bias:.4f} based on positive rate of {pos_rate:.3f}")

    # Use AdamW optimizer
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=1e-4)
    # FinalMLP safe loading
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        try:
            final_model.load_state_dict(torch.load(pretrained_model_path))
            logger.info(f"✅ Loaded pretrained weights into FinalMLP from {pretrained_model_path}")
        except RuntimeError as e:
            logger.warning(f"⚠️ Could not load pretrained weights into FinalMLP due to architecture mismatch: {e}")
            logger.warning("Proceeding with randomly initialized weights.")
    
    # Use AdamW optimizer
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Use BCEWithLogitsLoss with capped pos_weight
    y_train_full_tensor = torch.tensor(y_train_full, dtype=torch.float32)
    pos_count = float((y_train_full_tensor == 1).sum())
    neg_count = float((y_train_full_tensor == 0).sum())
    pos_weight = min(neg_count / pos_count, 2.0)  # Cap at 2.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    
    # Simple DataLoader without WeightedRandomSampler
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # === HISTORY TRACKING WITH EARLY STOPPING ===
    from sklearn.metrics import f1_score
    
    # Use ReduceLROnPlateau based on validation loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    patience, best_metric, best_state = 10, -1, None
    epochs_no_improve = 0
    
    train_losses, val_losses, val_accs, val_f1s = [], [], [], []
    
    for epoch in range(max_epochs):
        final_model.train()
        batch_train_losses = []
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            out = final_model(batch_X)
            out = out.view(-1)
            batch_y = batch_y.view(-1)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            batch_train_losses.append(loss.item())
        train_losses.append(sum(batch_train_losses) / len(batch_train_losses))

        # Validation
        final_model.eval()
        val_loss_total = 0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                out = final_model(batch_X)
                out = out.view(-1)
                batch_y = batch_y.view(-1)
                loss = criterion(out, batch_y)
                val_loss_total += loss.item()
                
                # Collect predictions for F1 calculation - ensure sigmoid is applied
                probs = torch.sigmoid(out)
                preds = (probs > 0.5).float()
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(batch_y.cpu().numpy())
        
        val_loss = val_loss_total / len(val_loader)
        val_acc = sum(p == l for p, l in zip(all_val_preds, all_val_labels)) / len(all_val_labels) if all_val_labels else 0
        
        # Calculate F1 score for minority class
        try:
            val_f1 = f1_score(all_val_labels, all_val_preds, pos_label=1)
        except:
            val_f1 = 0.0
            
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        
        # Early stopping based on F1 score
        scheduler.step(val_loss)  # Use validation loss for scheduler
        if val_f1 > best_metric:
            best_metric = val_f1
            best_state = final_model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Restore best weights
    if best_state:
        final_model.load_state_dict(best_state)

    # Check for underfitting
    if val_acc < 0.55:
        logger.warning(f"⚠️ Validation accuracy ({val_acc:.3f}) is below 0.55 - model may not be learning signal effectively")
        logger.warning("Consider: increasing model capacity, adjusting learning rate, or checking data quality")

    # === Find best threshold on validation set ===
    final_model.eval()
    val_probs = []
    val_labels = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            out = final_model(batch_X)
            probs = torch.sigmoid(out.view(-1))  # Ensure sigmoid is applied
            val_probs.extend(probs.cpu().numpy())
            val_labels.extend(batch_y.view(-1).cpu().numpy())
    
    best_threshold, best_threshold_score = find_best_threshold(val_labels, val_probs, metric='f1')
    
    # Log final model performance
    logger.info(f"Final model performance:")
    logger.info(f"  - Validation accuracy: {val_acc:.4f}")
    logger.info(f"  - Validation F1: {val_f1:.4f}")
    logger.info(f"  - Best threshold: {best_threshold:.4f}")
    logger.info(f"  - Threshold F1 score: {best_threshold_score:.4f}")
    
    # === Save scaler for consistent inference ===
    import joblib
    scaler_path = 'models/saved/nn_scaler.pkl'
    try:
        joblib.dump(full_scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
    except Exception as e:
        logger.warning(f"Failed to save scaler: {e}")
    
    # === Return everything for visualizations ===
    history = {
        "train_loss": train_losses, 
        "val_loss": val_losses, 
        "val_acc": val_accs,
        "val_f1": val_f1s,
        "best_threshold": best_threshold,
        "best_threshold_score": best_threshold_score,
        "epochs": len(train_losses)
    }
    
    # After best trial, save params if requested
    if params_save_path is not None:
        import json
        with open(params_save_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        logger.info(f"Saved best architecture params to {params_save_path}")
    
    return final_model, full_scaler, best_params, history


# Optional: quick sanity test
if __name__ == "__main__":
    mlp = MLP(input_dim=12, output_dim=1)
    x = torch.rand((4, 12))
    print("MLP logits:", mlp(x))

    lstm = LSTMNet(input_dim=5, output_dim=3)
    seq = torch.rand((4, 20, 5))
    print("LSTM logits:", lstm(seq))
