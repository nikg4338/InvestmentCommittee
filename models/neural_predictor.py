# Neural network predictor module for Investment Committee
# PyTorch-based MLP and LSTM models for price direction prediction

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import math
import os

# PyTorch imports (with fallback for testing)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available - using dummy implementations")

logger = logging.getLogger(__name__)

# Define classes based on PyTorch availability
if PYTORCH_AVAILABLE:
    class MLPPredictor(nn.Module):
        """
        Multi-Layer Perceptron for price direction prediction.
        Suitable for structured/tabular data with technical indicators.
        """
        
        def __init__(self, input_size: int = 12, hidden_sizes: List[int] = [64, 32, 16], 
                     output_size: int = 1, dropout_rate: float = 0.2):
            """
            Initialize MLP predictor.
            
            Args:
                input_size (int): Number of input features
                hidden_sizes (List[int]): Sizes of hidden layers
                output_size (int): Number of output classes (1 for binary classification)
                dropout_rate (float): Dropout rate for regularization
            """
            super(MLPPredictor, self).__init__()
            
            # Build layers
            layers = []
            prev_size = input_size
            
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                prev_size = hidden_size
            
            # Output layer (for binary classification, use single output)
            layers.append(nn.Linear(prev_size, output_size))
            # No activation here - will use BCEWithLogitsLoss
            
            self.network = nn.Sequential(*layers)
            
            # Initialize weights
            self._initialize_weights()
        
        def _initialize_weights(self):
            """Initialize network weights."""
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
        
        def forward(self, x):
            """Forward pass through the network."""
            return self.network(x)

    class LSTMPredictor(nn.Module):
        """
        LSTM model for sequential price prediction.
        Suitable for time series data with historical price sequences.
        """
        
        def __init__(self, input_size: int = 5, hidden_size: int = 64, 
                     num_layers: int = 2, output_size: int = 1, dropout_rate: float = 0.2):
            """
            Initialize LSTM predictor.
            
            Args:
                input_size (int): Number of input features per time step
                hidden_size (int): Size of LSTM hidden state
                num_layers (int): Number of LSTM layers
                output_size (int): Number of output classes
                dropout_rate (float): Dropout rate for regularization
            """
            super(LSTMPredictor, self).__init__()
            
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            # LSTM layers
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout_rate if num_layers > 1 else 0,
                batch_first=True
            )
            
            # Fully connected layers
            self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
            self.dropout = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(hidden_size // 2, output_size)
            
            # Initialize weights
            self._initialize_weights()
        
        def _initialize_weights(self):
            """Initialize network weights."""
            for name, param in self.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        
        def forward(self, x):
            """Forward pass through the LSTM network."""
            # LSTM forward pass
            lstm_out, (h_n, c_n) = self.lstm(x)
            
            # Use the last output
            last_output = lstm_out[:, -1, :]
            
            # Fully connected layers
            out = F.relu(self.fc1(last_output))
            out = self.dropout(out)
            out = self.fc2(out)
            
            return out

else:
    # Dummy implementations when PyTorch is not available
    class MLPPredictor:
        def __init__(self, input_size: int = 12, hidden_sizes: List[int] = [64, 32, 16], 
                     output_size: int = 1, dropout_rate: float = 0.2):
            self.input_size = input_size
            self.hidden_sizes = hidden_sizes
            self.output_size = output_size
            self.dropout_rate = dropout_rate
            logger.warning("PyTorch not available - using dummy MLP implementation")
        
        def forward(self, x):
            batch_size = len(x) if hasattr(x, '__len__') else 1
            return np.random.random(batch_size).astype(np.float32)

    class LSTMPredictor:
        def __init__(self, input_size: int = 5, hidden_size: int = 64, 
                     num_layers: int = 2, output_size: int = 1, dropout_rate: float = 0.2):
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.output_size = output_size
            self.dropout_rate = dropout_rate
            logger.warning("PyTorch not available - using dummy LSTM implementation")
        
        def forward(self, x):
            batch_size = len(x) if hasattr(x, '__len__') else 1
            return np.random.random(batch_size).astype(np.float32)


class NeuralPredictor:
    """
    Neural network predictor orchestrator.
    Manages both MLP and LSTM models for price direction prediction.
    """
    
    def __init__(self, model_type: str = 'mlp', model_path: Optional[str] = None):
        """
        Initialize neural predictor.
        
        Args:
            model_type (str): Type of model ('mlp' or 'lstm')
            model_path (str, optional): Path to saved model weights
        """
        self.model_type = model_type.lower()
        self.model_path = model_path
        self.model = None
        self.optimizer = None
        self.scaler = None  # For feature scaling
        self.best_threshold = 0.5  # Default threshold
        self.is_trained = False
        
        # Set device for PyTorch
        if PYTORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None
        
        # Try to load saved scaler
        self._load_scaler()
        
        # Initialize model (will be properly initialized during training)
        if self.model_type == 'mlp':
            self.model = None  # Will be created in train() with correct input size
        elif self.model_type == 'lstm':
            self.model = LSTMPredictor()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Set up optimizer (will be set during training)
        self.optimizer = None
        
        logger.info(f"Neural predictor initialized with {model_type.upper()} model")
    
    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict probability scores for class 1 using the trained model.
        Returns an array of probabilities between 0 and 1.
        
        Args:
            X: Input features (numpy array or PyTorch tensor)
            
        Returns:
            np.ndarray: Probability scores for class 1
        """
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained - returning dummy probabilities")
            return np.random.random(len(X) if hasattr(X, '__len__') else 1)
        
        self.model.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.tensor(X, dtype=torch.float32).to(self.device)
            elif not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32).to(self.device)
            
            logits = self.model(X)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
            # Add safety check for collapsed predictions
            if np.allclose(probs, probs[0]):
                logger.warning("Neural net probabilities are nearly constant. Stacking signal may be weak.")
            
            return probs
    
    def _load_scaler(self):
        """Load saved scaler and threshold for consistent inference."""
        try:
            import joblib
            import json
            
            # Load scaler
            scaler_path = 'models/saved/nn_scaler.pkl'
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Loaded saved scaler from {scaler_path}")
            else:
                logger.info("No saved scaler found, will create during training")
            
            # Load best threshold
            threshold_path = 'models/saved/nn_threshold.json'
            if os.path.exists(threshold_path):
                with open(threshold_path, 'r') as f:
                    threshold_info = json.load(f)
                    self.best_threshold = threshold_info.get('best_threshold', 0.5)
                    logger.info(f"Loaded best threshold: {self.best_threshold:.3f}")
            else:
                logger.info("No saved threshold found, using default 0.5")
                
        except Exception as e:
            logger.warning(f"Failed to load scaler/threshold: {e}")
            self.scaler = None
            self.best_threshold = 0.5
    
    def predict_nn_signal(self, features: Dict[str, Any]) -> Tuple[str, float, Dict[str, Any]]:
        """
        Predict price direction using neural network.
        
        Args:
            features (Dict[str, Any]): Input features
                For MLP: {'technicals': Dict[str, float]}
                For LSTM: {'sequence': List[List[float]], 'technicals': Dict[str, float]}
        
        Returns:
            Tuple[str, float, Dict[str, Any]]: (direction, confidence, metadata)
                - direction: 'BULLISH', 'BEARISH', or 'NEUTRAL'
                - confidence: Confidence score from 0.0 to 1.0
                - metadata: Additional prediction information
        """
        try:
            # Prepare input data
            if self.model_type == 'mlp':
                input_data = self._prepare_mlp_input(features)
            elif self.model_type == 'lstm':
                input_data = self._prepare_lstm_input(features)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            # Make prediction
            if PYTORCH_AVAILABLE and self.model:
                self.model.eval()
                with torch.no_grad():
                    if self.model_type == 'mlp':
                        predictions = self.model(input_data)
                    else:  # LSTM
                        predictions = self.model(input_data)
                    
                    # Convert to numpy for easier processing
                    if hasattr(predictions, 'numpy'):
                        probs = predictions.numpy()[0]
                    else:
                        probs = predictions[0]
            else:
                # Dummy prediction logic
                probs = self._dummy_neural_prediction(features)
            
            # Interpret results
            direction, confidence = self._interpret_neural_output(probs)
            
            # Create metadata
            metadata = {
                'model_type': self.model_type,
                'prediction_time': datetime.now().isoformat(),
                'probabilities': {
                    'bullish': float(probs[0]),
                    'neutral': float(probs[1]),
                    'bearish': float(probs[2])
                },
                'is_trained': self.is_trained,
                'pytorch_available': PYTORCH_AVAILABLE
            }
            
            logger.info(f"Neural prediction: {direction} with {confidence:.3f} confidence")
            return direction, confidence, metadata
            
        except Exception as e:
            logger.error(f"Error in neural prediction: {e}")
            return 'NEUTRAL', 0.5, {'error': str(e)}
    
    def _prepare_mlp_input(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Prepare input data for MLP model with consistent scaling.
        
        Args:
            features (Dict[str, Any]): Input features
            
        Returns:
            np.ndarray: Prepared input data
        """
        technicals = features.get('technicals', {})
        
        # Standard technical indicators
        input_features = [
            technicals.get('rsi', 50.0) / 100.0,  # Normalize RSI
            technicals.get('macd_signal', 0.0),
            technicals.get('bollinger_position', 0.5),
            technicals.get('volume_ratio', 1.0) / 5.0,  # Normalize volume ratio
            technicals.get('price_momentum', 0.0),
            technicals.get('volatility_rank', 50.0) / 100.0,  # Normalize IV rank
            technicals.get('vix_level', 20.0) / 100.0,  # Normalize VIX
            technicals.get('market_trend', 0.0),
            # Additional derived features
            technicals.get('price_volatility', 0.0),
            technicals.get('support_distance', 0.0),
            technicals.get('resistance_distance', 0.0),
            technicals.get('trend_strength', 0.0)
        ]
        
        # Apply scaler if available
        input_array = np.array([input_features])
        if self.scaler is not None:
            try:
                input_array = self.scaler.transform(input_array)
            except Exception as e:
                logger.warning(f"Failed to apply scaler: {e}")
        
        if PYTORCH_AVAILABLE:
            return torch.tensor(input_array, dtype=torch.float32)
        else:
            return input_array
    
    def _prepare_lstm_input(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Prepare input data for LSTM model.
        
        Args:
            features (Dict[str, Any]): Input features
            
        Returns:
            np.ndarray: Prepared sequence data
        """
        sequence = features.get('sequence', [])
        
        # If no sequence provided, create dummy sequence
        if not sequence:
            # Create a dummy sequence of 20 time steps
            sequence = [[100.0 + i * 0.5, 1000000, 100.0 + i * 0.5, 100.0 + i * 0.5, 100.0 + i * 0.5] 
                       for i in range(20)]
        
        # Normalize sequence data
        normalized_sequence = []
        for timestep in sequence:
            if len(timestep) >= 5:
                normalized_timestep = [
                    timestep[0] / 200.0,  # Normalize price (assuming ~$200 max)
                    timestep[1] / 100000000.0,  # Normalize volume
                    timestep[2] / 200.0,  # Normalize high
                    timestep[3] / 200.0,  # Normalize low
                    timestep[4] / 200.0   # Normalize close
                ]
                normalized_sequence.append(normalized_timestep)
        
        # Pad or truncate to fixed length (20 timesteps)
        target_length = 20
        if len(normalized_sequence) < target_length:
            # Pad with last value
            last_value = normalized_sequence[-1] if normalized_sequence else [0.5, 0.5, 0.5, 0.5, 0.5]
            normalized_sequence.extend([last_value] * (target_length - len(normalized_sequence)))
        elif len(normalized_sequence) > target_length:
            # Take last 20 values
            normalized_sequence = normalized_sequence[-target_length:]
        
        if PYTORCH_AVAILABLE:
            return torch.tensor([normalized_sequence], dtype=torch.float32)
        else:
            return np.array([normalized_sequence])
    
    def _dummy_neural_prediction(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Generate dummy neural network prediction.
        
        Args:
            features (Dict[str, Any]): Input features
            
        Returns:
            np.ndarray: Dummy prediction probabilities
        """
        # Use some heuristics to generate reasonable predictions
        technicals = features.get('technicals', {})
        
        rsi = technicals.get('rsi', 50.0)
        vix = technicals.get('vix_level', 20.0)
        momentum = technicals.get('price_momentum', 0.0)
        
        # Base probabilities
        bullish_prob = 0.33
        neutral_prob = 0.34
        bearish_prob = 0.33
        
        # Adjust based on indicators
        if rsi < 30:  # Oversold
            bullish_prob += 0.2
            bearish_prob -= 0.1
        elif rsi > 70:  # Overbought
            bearish_prob += 0.2
            bullish_prob -= 0.1
        
        if vix < 15:  # Low volatility
            bullish_prob += 0.1
            bearish_prob -= 0.05
        elif vix > 30:  # High volatility
            bearish_prob += 0.15
            bullish_prob -= 0.1
        
        if momentum > 0.3:  # Strong positive momentum
            bullish_prob += 0.15
            bearish_prob -= 0.1
        elif momentum < -0.3:  # Strong negative momentum
            bearish_prob += 0.15
            bullish_prob -= 0.1
        
        # Normalize probabilities
        total = bullish_prob + neutral_prob + bearish_prob
        probs = np.array([bullish_prob / total, neutral_prob / total, bearish_prob / total])
        
        return probs
    
    def _interpret_neural_output(self, probs: np.ndarray) -> Tuple[str, float]:
        """
        Interpret neural network output probabilities.
        
        Args:
            probs (np.ndarray): Prediction probabilities [bullish, neutral, bearish]
            
        Returns:
            Tuple[str, float]: (direction, confidence)
        """
        directions = ['BULLISH', 'NEUTRAL', 'BEARISH']
        max_idx = np.argmax(probs)
        direction = directions[max_idx]
        confidence = float(probs[max_idx])
        
        return direction, confidence
    
    def evaluate_predictions(self, y_true, y_pred, y_probs=None):
        """
        Evaluate predictions with guards for confidence collapse.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_probs: Predicted probabilities (optional)
            
        Returns:
            Dict: Evaluation metrics
        """
        from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_recall_curve, auc
        
        # Guard against all-one predictions (confidence collapse)
        unique_preds = np.unique(y_pred)
        if len(unique_preds) == 1:
            logger.warning(f"⚠️ CONFIDENCE COLLAPSE DETECTED: All predictions are {unique_preds[0]}")
            logger.warning("This indicates the model is not learning meaningful patterns")
            logger.warning("Consider: checking data quality, adjusting learning rate, or increasing model capacity")
        
        # Check for extreme class imbalance in predictions
        pred_counts = np.bincount(y_pred.astype(int))
        if len(pred_counts) > 1:
            pred_ratio = max(pred_counts) / min(pred_counts)
            if pred_ratio > 10:
                logger.warning(f"⚠️ Extreme prediction imbalance: {pred_ratio:.1f}:1 ratio")
        
        # Calculate metrics
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        
        # Calculate PR-AUC if probabilities are provided
        pr_auc = None
        if y_probs is not None:
            try:
                precision, recall, _ = precision_recall_curve(y_true, y_probs)
                pr_auc = auc(recall, precision)
            except:
                logger.warning("Could not calculate PR-AUC")
        
        return {
            'confusion_matrix': cm,
            'classification_report': report,
            'f1_score': f1,
            'pr_auc': pr_auc,
            'prediction_distribution': pred_counts.tolist() if len(pred_counts) > 0 else [0]
        }
    
    def train(self, X, y, epochs: int = 100, batch_size: int = 32, validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the neural network model.
        
        Args:
            X: Input features (pd.DataFrame or numpy array)
            y: Target labels (pd.Series or numpy array) - binary classification
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Proportion of data to use for validation
            
        Returns:
            Dict[str, Any]: Training results
        """
        if not PYTORCH_AVAILABLE:
            logger.warning("PyTorch not available - cannot train model")
            return {'error': 'PyTorch not available'}
        
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            
            # Convert inputs to numpy arrays if needed
            if hasattr(X, 'values'):
                X = X.values
            if hasattr(y, 'values'):
                y = y.values
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
            
            # Standardize features for MLP
            if self.model_type == 'mlp':
                self.scaler = StandardScaler()
                X_train = self.scaler.fit_transform(X_train)
                X_val = self.scaler.transform(X_val)
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train)
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Initialize model with correct input size
            if self.model_type == 'mlp':
                input_size = X_train.shape[1]
                if PYTORCH_AVAILABLE:
                    self.model = MLPPredictor(input_size=input_size, output_size=1)  # Binary classification
                else:
                    return {'error': 'PyTorch not available'}
            else:
                # LSTM model setup would go here
                logger.warning("LSTM training not fully implemented yet")
                return {'error': 'LSTM training not implemented'}
            
            # Setup loss and optimizer
            criterion = nn.BCEWithLogitsLoss()  # Better for binary classification
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
            # Training loop
            train_losses = []
            val_losses = []
            val_accuracies = []
            
            logger.info(f"Starting training for {epochs} epochs...")
            
            for epoch in range(epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()
                
                # Validation phase
                self.model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X).squeeze()
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        
                        # Calculate accuracy
                        predicted = (torch.sigmoid(outputs) > 0.5).float()
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()
                
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                val_accuracy = correct / total
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Mark as trained
            self.is_trained = True
            
            logger.info("Training completed successfully")
            
            return {
                'epochs': epochs,
                'samples': len(X),
                'model_type': self.model_type,
                'status': 'completed',
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1],
                'final_val_accuracy': val_accuracies[-1],
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies
            }
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    def save_model(self, path: str) -> bool:
        """
        Save model weights to file.
        
        Args:
            path (str): Path to save model
            
        Returns:
            bool: True if successful
        """
        try:
            if PYTORCH_AVAILABLE and self.model:
                import torch
                torch.save(self.model.state_dict(), path)
                logger.info(f"Model saved to {path}")
                return True
            else:
                logger.warning("Cannot save model - PyTorch not available")
                return False
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """
        Load model weights from file.
        
        Args:
            path (str): Path to load model from
            
        Returns:
            bool: True if successful
        """
        try:
            if PYTORCH_AVAILABLE and self.model and os.path.exists(path):
                import torch
                self.model.load_state_dict(torch.load(path))
                self.is_trained = True
                logger.info(f"Model loaded from {path}")
                return True
            else:
                logger.warning(f"Cannot load model from {path}")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    def extract_features(self, X: torch.Tensor) -> np.ndarray:
        """
        Extracts intermediate hidden layer features for stacking.
        Assumes self.model has a .model attribute that's an nn.Sequential.
        """
        self.model.eval()
        with torch.no_grad():
            # Example: up to hidden2 ReLU + Dropout (update if your architecture is different)
            x = self.model.model[0](X)   # Linear1
            x = self.model.model[1](x)   # ReLU1
            x = self.model.model[2](x)   # Dropout1
            x = self.model.model[3](x)   # Linear2
            x = self.model.model[4](x)   # ReLU2
            x = self.model.model[5](x)   # Dropout2
            return x.cpu().numpy()

# Convenience functions
def predict_nn_signal(features: Dict[str, Any], model_type: str = 'mlp') -> Tuple[str, float, Dict[str, Any]]:
    """
    Convenience function for neural network prediction.
    
    Args:
        features (Dict[str, Any]): Input features
        model_type (str): Type of model to use ('mlp' or 'lstm')
        
    Returns:
        Tuple[str, float, Dict[str, Any]]: (direction, confidence, metadata)
    """
    predictor = NeuralPredictor(model_type=model_type)
    return predictor.predict_nn_signal(features)


def create_sample_neural_features() -> Dict[str, Any]:
    """
    Create sample features for neural network testing.
    
    Returns:
        Dict[str, Any]: Sample features for both MLP and LSTM
    """
    technicals = {
        'rsi': 45.0,
        'macd_signal': 0.2,
        'bollinger_position': 0.4,
        'volume_ratio': 1.2,
        'price_momentum': 0.1,
        'volatility_rank': 50.0,
        'vix_level': 18.0,
        'market_trend': 0.3,
        'price_volatility': 0.02,
        'support_distance': 0.05,
        'resistance_distance': 0.03,
        'trend_strength': 0.15
    }
    
    # Sample sequence data for LSTM (20 timesteps of OHLCV)
    sequence = []
    base_price = 150.0
    for i in range(20):
        price = base_price + i * 0.5 + np.random.normal(0, 0.5)
        volume = 50000000 + np.random.normal(0, 5000000)
        high = price + abs(np.random.normal(0, 1))
        low = price - abs(np.random.normal(0, 1))
        close = price + np.random.normal(0, 0.5)
        sequence.append([price, volume, high, low, close])
    
    return {
        'technicals': technicals,
        'sequence': sequence
    }
def extract_features(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Extracts intermediate features from the first hidden layer of the neural net for stacking.
    """
    self.model.eval()
    with torch.no_grad():
        # Ensure X is a torch tensor on the correct device
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
        elif not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
        # Pass through only up to the first hidden layer (assuming MLP)
        hidden = self.model.network[0](X)       # Linear
        hidden = self.model.network[1](hidden)  # ReLU
        hidden = self.model.network[2](hidden)  # Dropout
        return hidden.cpu().numpy()


def test_neural_predictor():
    """Test the neural predictor with sample data."""
    print("Testing Neural Predictor...")
    print(f"PyTorch Available: {PYTORCH_AVAILABLE}")
    
    # Create sample features
    features = create_sample_neural_features()
    
    # Test MLP
    print("\n=== Testing MLP ===")
    direction, confidence, metadata = predict_nn_signal(features, model_type='mlp')
    print(f"Direction: {direction}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Probabilities: {metadata['probabilities']}")
    
    # Test LSTM
    print("\n=== Testing LSTM ===")
    direction, confidence, metadata = predict_nn_signal(features, model_type='lstm')
    print(f"Direction: {direction}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Probabilities: {metadata['probabilities']}")
    
    # Test different scenarios
    print("\n=== Testing Scenarios ===")
    
    # Bullish scenario
    bullish_features = features.copy()
    bullish_features['technicals'].update({
        'rsi': 25.0,  # Oversold
        'vix_level': 12.0,  # Low VIX
        'price_momentum': 0.6  # Strong momentum
    })
    direction, confidence, metadata = predict_nn_signal(bullish_features, model_type='mlp')
    print(f"Bullish scenario: {direction} ({confidence:.3f})")
    
    # Bearish scenario
    bearish_features = features.copy()
    bearish_features['technicals'].update({
        'rsi': 80.0,  # Overbought
        'vix_level': 35.0,  # High VIX
        'price_momentum': -0.6  # Negative momentum
    })
    direction, confidence, metadata = predict_nn_signal(bearish_features, model_type='mlp')
    print(f"Bearish scenario: {direction} ({confidence:.3f})")


if __name__ == "__main__":
    test_neural_predictor() 