#!/usr/bin/env python3
"""
Advanced Model Training Pipeline
===============================

Comprehensive model training pipeline that addresses all identified issues:
- Standardized feature ordering
- Proper model calibration  
- Advanced ensemble techniques
- Uncertainty quantification
- Performance monitoring
"""

import json
import joblib
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV, IsotonicRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')


def json_serializable(obj):
    """Convert numpy types to JSON serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [json_serializable(item) for item in obj]
    return obj

# ML Models
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
import lightgbm as lgb
import xgboost as xgb

# Neural Networks
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedModelTrainer:
    """Advanced ML model trainer with all modern best practices."""
    
    def __init__(self, feature_manifest_path: str = "models/feature_order_manifest.json"):
        """Initialize trainer with feature ordering manifest."""
        self.feature_manifest_path = feature_manifest_path
        self.feature_order = self._load_feature_order()
        self.models = {}
        self.calibrated_models = {}
        self.preprocessing_pipelines = {}
        self.feature_importances = {}
        self.training_metrics = {}
        
        # Class imbalance handling configuration
        self.imbalance_config = {
            'use_smote': True,
            'smote_strategy': 'BorderlineSMOTE',  # BorderlineSMOTE, ADASYN, SMOTE
            'use_class_weights': True,
            'positive_threshold': 0.35,  # Lower threshold for imbalanced data
            'use_ensemble_resampling': True
        }
        
        # Cross-validation configuration
        self.cv_config = {
            'use_time_series_cv': True,
            'n_splits': 5,
            'test_size': 0.2,
            'validation_size': 0.15
        }
        
        logger.info(f"✅ Advanced Model Trainer initialized")
        logger.info(f"   Feature order loaded: {len(self.feature_order)} features")
        logger.info(f"   Class imbalance handling: {self.imbalance_config}")
        logger.info(f"   Cross-validation: {self.cv_config}")
        
    def _load_feature_order(self) -> List[str]:
        """Load canonical feature ordering."""
        try:
            with open(self.feature_manifest_path, 'r') as f:
                manifest = json.load(f)
            return manifest['feature_order']
        except Exception as e:
            logger.error(f"❌ Failed to load feature manifest: {e}")
            raise
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'target_1d_enhanced') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data with exact feature ordering and validation."""
        logger.info(f"📊 Preparing data for training...")
        
        # Ensure we have the target column
        if target_column not in df.columns:
            logger.error(f"❌ Target column '{target_column}' not found in data")
            raise ValueError(f"Target column '{target_column}' not found")
        
        # Extract features in exact order
        available_features = [f for f in self.feature_order if f in df.columns]
        missing_features = [f for f in self.feature_order if f not in df.columns]
        
        if missing_features:
            logger.warning(f"⚠️ Missing features: {len(missing_features)}")
            logger.warning(f"   First 10 missing: {missing_features[:10]}")
        
        logger.info(f"✅ Using {len(available_features)}/{len(self.feature_order)} features")
        
        # Create feature matrix with exact ordering
        X = df[available_features].copy()
        y = df[target_column].copy()
        
        # Handle missing values and infinite values
        X = X.fillna(0.0)  # Simple imputation for now
        X = X.replace([np.inf, -np.inf], 0.0)  # Replace infinite values
        
        # Remove samples with NaN targets
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"📊 Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"   Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def create_preprocessing_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """Create standardized preprocessing pipeline."""
        logger.info(f"🔧 Creating preprocessing pipeline...")
        
        # Identify numeric features (all features in our case)
        numeric_features = X.columns.tolist()
        
        # Create preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', RobustScaler(), numeric_features)  # RobustScaler is less sensitive to outliers
            ],
            remainder='drop'
        )
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor)
        ])
        
        logger.info(f"✅ Preprocessing pipeline created for {len(numeric_features)} features")
        return pipeline
    
    def handle_class_imbalance(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance using SMOTE and other techniques."""
        logger.info(f"⚖️ Handling class imbalance...")
        
        # Convert y_train to pandas Series if it's numpy array
        if isinstance(y_train, np.ndarray):
            y_train = pd.Series(y_train)
        
        # Check current class distribution
        class_counts = y_train.value_counts()
        minority_ratio = class_counts.min() / class_counts.max()
        logger.info(f"   Original distribution: {dict(class_counts)}")
        logger.info(f"   Minority ratio: {minority_ratio:.3f}")
        
        # Only apply SMOTE if imbalance is severe
        if minority_ratio < 0.3 and self.imbalance_config['use_smote']:
            try:
                # Choose SMOTE strategy
                if self.imbalance_config['smote_strategy'] == 'BorderlineSMOTE':
                    smote = BorderlineSMOTE(random_state=42, k_neighbors=3)
                elif self.imbalance_config['smote_strategy'] == 'ADASYN':
                    smote = ADASYN(random_state=42, n_neighbors=3)
                else:
                    smote = SMOTE(random_state=42, k_neighbors=3)
                
                # Apply SMOTE
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                
                # Convert back to DataFrame/Series
                X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
                y_resampled = pd.Series(y_resampled, name=y_train.name)
                
                new_class_counts = y_resampled.value_counts()
                logger.info(f"   After SMOTE: {dict(new_class_counts)}")
                logger.info(f"   New minority ratio: {new_class_counts.min() / new_class_counts.max():.3f}")
                
                return X_resampled, y_resampled
                
            except Exception as e:
                logger.warning(f"⚠️ SMOTE failed: {e}, using original data")
                return X_train, y_train
        else:
            logger.info(f"   Skipping SMOTE (ratio: {minority_ratio:.3f})")
            return X_train, y_train
    
    def get_class_weights(self, y: pd.Series) -> Dict[int, float]:
        """Calculate class weights for imbalanced data."""
        if self.imbalance_config['use_class_weights']:
            try:
                classes = np.unique(y)
                weights = compute_class_weight('balanced', classes=classes, y=y)
                class_weights = dict(zip(classes, weights))
                logger.info(f"📊 Class weights: {class_weights}")
                return class_weights
            except Exception as e:
                logger.warning(f"⚠️ Failed to compute class weights: {e}")
                return None
        return None
    
    def setup_time_series_cv(self, X: pd.DataFrame) -> TimeSeriesSplit:
        """Setup time series cross-validation."""
        if self.cv_config['use_time_series_cv']:
            tscv = TimeSeriesSplit(
                n_splits=self.cv_config['n_splits'],
                test_size=int(len(X) * 0.15),  # 15% for each test fold
                gap=0  # No gap between train and test
            )
            logger.info(f"⏰ Time series CV setup: {self.cv_config['n_splits']} splits")
            return tscv
        return None
    
    def feature_selection(self, X: pd.DataFrame, y: pd.Series, max_features: int = 50) -> List[str]:
        """Perform feature selection to reduce dimensionality."""
        logger.info(f"🎯 Performing feature selection...")
        
        # Use LightGBM for feature selection (fast and effective)
        selector_model = LGBMClassifier(
            n_estimators=100,
            num_leaves=31,
            importance_type='gain',
            random_state=42,
            verbose=-1
        )
        
        # Fit selector model
        selector_model.fit(X, y)
        
        # Get feature importances
        importances = selector_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Select top features
        selected_features = feature_importance_df.head(max_features)['feature'].tolist()
        
        logger.info(f"✅ Selected {len(selected_features)} features")
        logger.info(f"   Top 5: {selected_features[:5]}")
        
        # Save feature importance for analysis
        self.feature_importances['selection'] = feature_importance_df
        
        return selected_features
    
    def train_catboost_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                           X_val: pd.DataFrame, y_val: pd.Series) -> CatBoostClassifier:
        """Train optimized CatBoost model with class imbalance handling."""
        logger.info(f"🚀 Training CatBoost model...")
        
        # Get class weights
        class_weights = self.get_class_weights(y_train)
        
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            loss_function='Logloss',
            bootstrap_type='Bayesian',
            bagging_temperature=1,
            od_type='Iter',
            od_wait=50,
            random_seed=42,
            verbose=100,
            class_weights=class_weights if class_weights else None,
            auto_class_weights='Balanced' if not class_weights else None
        )
        
        # Fit with validation set for early stopping
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=100
        )
        
        logger.info(f"✅ CatBoost training complete")
        return model
    
    def train_random_forest_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        """Train optimized Random Forest model with class imbalance handling."""
        logger.info(f"🌳 Training Random Forest model...")
        
        # Get class weights
        class_weights = self.get_class_weights(y_train)
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced' if not class_weights else class_weights
        )
        
        model.fit(X_train, y_train)
        
        logger.info(f"✅ Random Forest training complete")
        logger.info(f"   OOB Score: {model.oob_score_:.4f}")
        return model
    
    def train_svm_model(self, X_train_scaled: np.ndarray, y_train: pd.Series) -> SVC:
        """Train optimized SVM model with class imbalance handling."""
        logger.info(f"⚡ Training SVM model...")
        
        # Get class weights
        class_weights = self.get_class_weights(y_train)
        
        model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,  # Enable probability predictions
            random_state=42,
            class_weight='balanced' if not class_weights else class_weights
        )
        
        model.fit(X_train_scaled, y_train)
        
        logger.info(f"✅ SVM training complete")
        return model
    
    def train_lightgbm_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series) -> LGBMClassifier:
        """Train optimized LightGBM model with class imbalance handling."""
        logger.info(f"💫 Training LightGBM model...")
        
        # Get class weights
        class_weights = self.get_class_weights(y_train)
        
        model = LGBMClassifier(
            n_estimators=1000,
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            min_child_samples=20,
            random_state=42,
            verbose=-1,
            class_weight='balanced' if not class_weights else class_weights
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(100)
            ]
        )
        
        logger.info(f"✅ LightGBM training complete")
        return model
    
    def train_xgboost_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series) -> xgb.XGBClassifier:
        """Train optimized XGBoost model with class imbalance handling."""
        logger.info(f"🚀 Training XGBoost model...")
        
        # Calculate scale_pos_weight for imbalanced data
        pos_count = (y_train == 1).sum()
        neg_count = (y_train == 0).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
        
        model = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        logger.info(f"✅ XGBoost training complete")
        return model
    
    def train_neural_network_model(self, X_train_scaled: np.ndarray, y_train: pd.Series,
                                 X_val_scaled: np.ndarray, y_val: pd.Series) -> Optional[Any]:
        """Train modern neural network model."""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("⚠️ TensorFlow not available, skipping neural network")
            return None
            
        logger.info(f"🧠 Training Neural Network model...")
        
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        # Train with early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True
        )
        
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        
        logger.info(f"✅ Neural Network training complete")
        return model
    
    def calibrate_model(self, model: Any, X_cal: np.ndarray, y_cal: pd.Series, model_name: str) -> Any:
        """Enhanced model calibration with multiple methods."""
        logger.info(f"🎯 Calibrating {model_name} model...")
        
        try:
            # Choose calibration method based on model type
            if 'neural' in model_name.lower():
                # Use Platt scaling for neural networks (better for small datasets)
                method = 'sigmoid'
            elif 'svm' in model_name.lower():
                # Use Platt scaling for SVM (originally designed for it)
                method = 'sigmoid'
            else:
                # Use isotonic regression for tree-based models (more flexible)
                method = 'isotonic'
            
            calibrated_model = CalibratedClassifierCV(
                model,
                method=method,
                cv='prefit'
            )
            calibrated_model.fit(X_cal, y_cal)
            
            logger.info(f"✅ {model_name} calibration complete (method: {method})")
            return calibrated_model
        except Exception as e:
            logger.warning(f"⚠️ Calibration failed for {model_name}: {e}")
            return model
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: pd.Series, model_name: str) -> Dict[str, float]:
        """Comprehensive model evaluation with configurable threshold."""
        logger.info(f"📊 Evaluating {model_name} model...")
        
        try:
            # Get predictions
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = model.predict(X_test)
            
            # Use configurable threshold for class imbalance
            threshold = self.imbalance_config['positive_threshold']
            y_pred = (y_pred_proba > threshold).astype(int)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(set(y_test)) > 1 else 0.5,
                'avg_confidence': np.mean(np.abs(y_pred_proba - 0.5) * 2),
                'threshold_used': threshold
            }
            
            logger.info(f"✅ {model_name} evaluation complete (threshold={threshold}):")
            for metric, value in metrics.items():
                if metric != 'threshold_used':
                    logger.info(f"   {metric}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed for {model_name}: {e}")
            return {'error': str(e)}
    
    def save_model_with_metadata(self, model: Any, model_name: str, metrics: Dict[str, float],
                                feature_order: List[str], preprocessing_pipeline: Pipeline) -> None:
        """Save model with comprehensive metadata."""
        logger.info(f"💾 Saving {model_name} model with metadata...")
        
        # Create model directory
        model_dir = Path(f"models/production")
        model_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = model_dir / f"enhanced_{model_name}.pkl"
        joblib.dump(model, model_path)
        
        # Save preprocessing pipeline
        pipeline_path = model_dir / f"enhanced_{model_name}_preprocessing.pkl"
        joblib.dump(preprocessing_pipeline, pipeline_path)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'training_date': datetime.now().isoformat(),
            'feature_order': feature_order,
            'num_features': len(feature_order),
            'metrics': metrics,
            'model_path': str(model_path),
            'preprocessing_path': str(pipeline_path),
            'feature_manifest_path': self.feature_manifest_path
        }
        
        metadata_path = model_dir / f"enhanced_{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(json_serializable(metadata), f, indent=2)
        
        logger.info(f"✅ {model_name} saved successfully")
        logger.info(f"   Model: {model_path}")
        logger.info(f"   Metadata: {metadata_path}")
    
    def train_all_models(self, df: pd.DataFrame, target_column: str = 'target_1d_enhanced',
                        test_size: float = 0.2, val_size: float = 0.1,
                        feature_selection: bool = True, max_features: int = 50) -> Dict[str, Any]:
        """Train all models with comprehensive pipeline."""
        logger.info(f"🚀 Starting comprehensive model training pipeline...")
        
        # Prepare data
        X, y = self.prepare_data(df, target_column)
        
        # Feature selection
        if feature_selection:
            selected_features = self.feature_selection(X, y, max_features)
            X = X[selected_features]
            feature_order = selected_features
        else:
            feature_order = X.columns.tolist()
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
        )
        
        # Further split for calibration
        X_train_fit, X_cal, y_train_fit, y_cal = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Handle class imbalance with SMOTE (applied only to training data)
        X_train_balanced, y_train_balanced = self.handle_class_imbalance(X_train_fit, y_train_fit)
        
        logger.info(f"📊 Data splits:")
        logger.info(f"   Train: {X_train_fit.shape[0]} samples")
        logger.info(f"   Train (balanced): {X_train_balanced.shape[0]} samples")
        logger.info(f"   Calibration: {X_cal.shape[0]} samples")
        logger.info(f"   Validation: {X_val.shape[0]} samples")
        logger.info(f"   Test: {X_test.shape[0]} samples")
        
        # Create preprocessing pipeline
        preprocessing_pipeline = self.create_preprocessing_pipeline(X_train_fit)
        preprocessing_pipeline.fit(X_train_fit)
        
        # Scale data for models that need it
        X_train_fit_scaled = preprocessing_pipeline.transform(X_train_fit)
        X_cal_scaled = preprocessing_pipeline.transform(X_cal)
        X_val_scaled = preprocessing_pipeline.transform(X_val)
        X_test_scaled = preprocessing_pipeline.transform(X_test)
        
        # Train models
        trained_models = {}
        all_metrics = {}
        
        # CatBoost (doesn't need scaling)
        try:
            catboost_model = self.train_catboost_model(X_train_balanced, y_train_balanced, X_val, y_val)
            catboost_calibrated = self.calibrate_model(catboost_model, X_cal, y_cal, 'catboost')
            catboost_metrics = self.evaluate_model(catboost_calibrated, X_test, y_test, 'catboost')
            
            trained_models['catboost'] = catboost_calibrated
            all_metrics['catboost'] = catboost_metrics
            
            self.save_model_with_metadata(
                catboost_calibrated, 'catboost', catboost_metrics, feature_order, preprocessing_pipeline
            )
        except Exception as e:
            logger.error(f"❌ CatBoost training failed: {e}")
        
        # Random Forest (doesn't need scaling)
        try:
            rf_model = self.train_random_forest_model(X_train_balanced, y_train_balanced)
            rf_calibrated = self.calibrate_model(rf_model, X_cal, y_cal, 'random_forest')
            rf_metrics = self.evaluate_model(rf_calibrated, X_test, y_test, 'random_forest')
            
            trained_models['random_forest'] = rf_calibrated
            all_metrics['random_forest'] = rf_metrics
            
            self.save_model_with_metadata(
                rf_calibrated, 'random_forest', rf_metrics, feature_order, preprocessing_pipeline
            )
        except Exception as e:
            logger.error(f"❌ Random Forest training failed: {e}")
        
        # SVM (needs scaling) - use balanced data
        X_train_balanced_scaled = preprocessing_pipeline.fit_transform(X_train_balanced)
        try:
            svm_model = self.train_svm_model(X_train_balanced_scaled, y_train_balanced)
            svm_calibrated = self.calibrate_model(svm_model, X_cal_scaled, y_cal, 'svm')
            svm_metrics = self.evaluate_model(svm_calibrated, X_test_scaled, y_test, 'svm')
            
            trained_models['svm'] = svm_calibrated
            all_metrics['svm'] = svm_metrics
            
            self.save_model_with_metadata(
                svm_calibrated, 'svm', svm_metrics, feature_order, preprocessing_pipeline
            )
        except Exception as e:
            logger.error(f"❌ SVM training failed: {e}")
        
        # LightGBM
        try:
            lgb_model = self.train_lightgbm_model(X_train_balanced, y_train_balanced, X_val, y_val)
            lgb_calibrated = self.calibrate_model(lgb_model, X_cal, y_cal, 'lightgbm')
            lgb_metrics = self.evaluate_model(lgb_calibrated, X_test, y_test, 'lightgbm')
            
            trained_models['lightgbm'] = lgb_calibrated
            all_metrics['lightgbm'] = lgb_metrics
            
            self.save_model_with_metadata(
                lgb_calibrated, 'lightgbm', lgb_metrics, feature_order, preprocessing_pipeline
            )
        except Exception as e:
            logger.error(f"❌ LightGBM training failed: {e}")
        
        # XGBoost
        try:
            xgb_model = self.train_xgboost_model(X_train_balanced, y_train_balanced, X_val, y_val)
            xgb_calibrated = self.calibrate_model(xgb_model, X_cal, y_cal, 'xgboost')
            xgb_metrics = self.evaluate_model(xgb_calibrated, X_test, y_test, 'xgboost')
            
            trained_models['xgboost'] = xgb_calibrated
            all_metrics['xgboost'] = xgb_metrics
            
            self.save_model_with_metadata(
                xgb_calibrated, 'xgboost', xgb_metrics, feature_order, preprocessing_pipeline
            )
        except Exception as e:
            logger.error(f"❌ XGBoost training failed: {e}")
        
        # Neural Network (needs scaling) - use balanced data
        X_train_balanced_scaled = preprocessing_pipeline.fit_transform(X_train_balanced)
        try:
            nn_model = self.train_neural_network_model(X_train_balanced_scaled, y_train_balanced, X_val_scaled, y_val)
            if nn_model is not None:
                nn_metrics = self.evaluate_model(nn_model, X_test_scaled, y_test, 'neural_network')
                
                trained_models['neural_network'] = nn_model
                all_metrics['neural_network'] = nn_metrics
                
                # Save neural network separately
                nn_path = Path("models/production/enhanced_neural_network.h5")
                nn_model.save(nn_path)
                
                metadata = {
                    'model_name': 'neural_network',
                    'training_date': datetime.now().isoformat(),
                    'feature_order': feature_order,
                    'metrics': nn_metrics,
                    'model_path': str(nn_path),
                    'preprocessing_path': str(Path("models/production/enhanced_neural_network_preprocessing.pkl"))
                }
                
                with open("models/production/enhanced_neural_network_metadata.json", 'w') as f:
                    json.dump(json_serializable(metadata), f, indent=2)
        except Exception as e:
            logger.error(f"❌ Neural Network training failed: {e}")
        
        # Save training summary
        summary = {
            'training_date': datetime.now().isoformat(),
            'total_samples': len(X),
            'total_features': X.shape[1],
            'feature_selection': feature_selection,
            'selected_features': feature_order,
            'models_trained': list(trained_models.keys()),
            'all_metrics': all_metrics,
            'best_model': max(all_metrics.keys(), key=lambda k: all_metrics[k].get('roc_auc', 0)) if all_metrics else None
        }
        
        with open("models/production/enhanced_training_summary.json", 'w') as f:
            json.dump(json_serializable(summary), f, indent=2)
        
        logger.info(f"🎉 Training pipeline complete!")
        logger.info(f"   Models trained: {len(trained_models)}")
        logger.info(f"   Best model (ROC-AUC): {summary.get('best_model', 'None')}")
        
        return {
            'models': trained_models,
            'metrics': all_metrics,
            'summary': summary,
            'feature_order': feature_order,
            'preprocessing_pipeline': preprocessing_pipeline
        }


def main():
    """Main training script."""
    logger.info("🚀 Starting Advanced Model Training Pipeline")
    
    # Initialize trainer
    trainer = AdvancedModelTrainer()
    
    # Load data (placeholder - replace with actual data loading)
    logger.info("📊 Loading training data...")
    
    # This would be replaced with actual data loading from data_collection_alpaca.py
    # For now, create a placeholder
    logger.warning("⚠️ Using placeholder data - replace with actual data loading")
    
    # Placeholder for actual implementation
    # from data_collection_alpaca import AlpacaDataCollector
    # collector = AlpacaDataCollector()
    # df = collector.collect_training_data([1, 2, 3])
    
    logger.info("✅ Training pipeline ready - load actual data to proceed")


if __name__ == "__main__":
    main()
