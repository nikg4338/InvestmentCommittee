
"""
Model Persistence Utilities
===========================

This module provides utilities for persisting and loading complete model artifacts
including trained models, calibrators, thresholds, and metadata for deployment.

Features:
- Model artifact serialization with joblib/pickle
- Threshold persistence for consistent deployment
- Calibrator storage for ensemble stability
- Metadata tracking (training date, config, performance)
- Version management for model updates
- Deployment-ready artifact packaging
"""

import logging
import pickle
import joblib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class ModelArtifactManager:
    """
    Manages complete model artifacts for consistent deployment.
    
    Handles storage and retrieval of:
    - Trained models
    - Probability calibrators
    - Operating thresholds
    - Feature metadata
    - Training configuration
    - Performance metrics
    """
    
    def __init__(self, artifacts_dir: str = "models/artifacts"):
        """
        Initialize the artifact manager.
        
        Args:
            artifacts_dir: Directory to store model artifacts
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📦 ModelArtifactManager initialized: {self.artifacts_dir}")
    
    def save_complete_model(self, 
                           model_name: str,
                           model: Any,
                           calibrator: Optional[Any] = None,
                           threshold: Optional[float] = None,
                           feature_names: Optional[List[str]] = None,
                           categorical_features: Optional[List[str]] = None,
                           metadata: Optional[Dict[str, Any]] = None,
                           version: str = "v1") -> str:
        """
        Save complete model artifact with all deployment components.
        
        Args:
            model_name: Name of the model
            model: Trained model object
            calibrator: Fitted calibrator for probability adjustment
            threshold: Operating threshold for binary classification
            feature_names: List of feature names in training order
            categorical_features: List of categorical feature names
            metadata: Additional metadata (config, performance, etc.)
            version: Model version identifier
            
        Returns:
            Path to saved artifact directory
        """
        # Create versioned artifact directory
        artifact_path = self.artifacts_dir / f"{model_name}_{version}"
        artifact_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare complete metadata
        complete_metadata = {
            'model_name': model_name,
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'model_type': type(model).__name__,
            'has_calibrator': calibrator is not None,
            'has_threshold': threshold is not None,
            'feature_count': len(feature_names) if feature_names else None,
            'categorical_count': len(categorical_features) if categorical_features else 0,
            'feature_names': feature_names,
            'categorical_features': categorical_features or [],
        }
        
        # Add user metadata
        if metadata:
            complete_metadata.update(metadata)
        
        try:
            # Save model using joblib (better for sklearn models)
            model_file = artifact_path / "model.joblib"
            joblib.dump(model, model_file)
            logger.info(f"✅ Saved model: {model_file}")
            
            # Save calibrator if present
            if calibrator is not None:
                calibrator_file = artifact_path / "calibrator.joblib"
                joblib.dump(calibrator, calibrator_file)
                logger.info(f"✅ Saved calibrator: {calibrator_file}")
            
            # Save threshold if present
            if threshold is not None:
                threshold_data = {
                    'threshold': float(threshold),
                    'threshold_type': 'binary_classification',
                    'saved_at': datetime.now().isoformat()
                }
                threshold_file = artifact_path / "threshold.json"
                with open(threshold_file, 'w') as f:
                    json.dump(threshold_data, f, indent=2)
                logger.info(f"✅ Saved threshold: {threshold} -> {threshold_file}")
            
            # Save metadata
            metadata_file = artifact_path / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(complete_metadata, f, indent=2, default=str)
            logger.info(f"✅ Saved metadata: {metadata_file}")
            
            # Create deployment manifest
            manifest = {
                'artifact_version': '1.0',
                'model_name': model_name,
                'version': version,
                'files': {
                    'model': 'model.joblib',
                    'metadata': 'metadata.json',
                    'calibrator': 'calibrator.joblib' if calibrator else None,
                    'threshold': 'threshold.json' if threshold else None
                },
                'deployment_ready': True,
                'created_at': datetime.now().isoformat()
            }
            
            manifest_file = artifact_path / "deployment_manifest.json"
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"🎯 Complete model artifact saved: {artifact_path}")
            return str(artifact_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save model artifact: {e}")
            raise
    
    def load_complete_model(self, model_name: str, version: str = "v1") -> Dict[str, Any]:
        """
        Load complete model artifact with all components.
        
        Args:
            model_name: Name of the model
            version: Model version to load
            
        Returns:
            Dictionary containing all model components
        """
        artifact_path = self.artifacts_dir / f"{model_name}_{version}"
        
        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
        
        try:
            # Load deployment manifest
            manifest_file = artifact_path / "deployment_manifest.json"
            if manifest_file.exists():
                with open(manifest_file, 'r') as f:
                    manifest = json.load(f)
                logger.info(f"📋 Loaded manifest for {model_name} {version}")
            else:
                manifest = {}
            
            # Load metadata
            metadata_file = artifact_path / "metadata.json"
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Load model
            model_file = artifact_path / "model.joblib"
            model = joblib.load(model_file)
            logger.info(f"✅ Loaded model: {model_file}")
            
            # Load calibrator if present
            calibrator = None
            calibrator_file = artifact_path / "calibrator.joblib"
            if calibrator_file.exists():
                calibrator = joblib.load(calibrator_file)
                logger.info(f"✅ Loaded calibrator: {calibrator_file}")
            
            # Load threshold if present
            threshold = None
            threshold_file = artifact_path / "threshold.json"
            if threshold_file.exists():
                with open(threshold_file, 'r') as f:
                    threshold_data = json.load(f)
                    threshold = threshold_data.get('threshold')
                logger.info(f"✅ Loaded threshold: {threshold}")
            
            # Prepare complete artifact
            artifact = {
                'model': model,
                'calibrator': calibrator,
                'threshold': threshold,
                'metadata': metadata,
                'manifest': manifest,
                'feature_names': metadata.get('feature_names'),
                'categorical_features': metadata.get('categorical_features', []),
                'model_name': model_name,
                'version': version,
                'artifact_path': str(artifact_path)
            }
            
            logger.info(f"🎯 Complete model artifact loaded: {model_name} {version}")
            return artifact
            
        except Exception as e:
            logger.error(f"❌ Failed to load model artifact: {e}")
            raise
    
    def save_ensemble_artifacts(self,
                               ensemble_name: str,
                               models: Dict[str, Any],
                               calibrators: Optional[Dict[str, Any]] = None,
                               thresholds: Optional[Dict[str, float]] = None,
                               ensemble_weights: Optional[Dict[str, float]] = None,
                               meta_model: Optional[Any] = None,
                               feature_info: Optional[Dict[str, Any]] = None,
                               version: str = "v1") -> str:
        """
        Save complete ensemble artifacts.
        
        Args:
            ensemble_name: Name of the ensemble
            models: Dictionary of trained models
            calibrators: Dictionary of calibrators per model
            thresholds: Dictionary of thresholds per model
            ensemble_weights: Model weights for ensemble voting
            meta_model: Meta-model for stacking
            feature_info: Feature metadata including categorical features
            version: Ensemble version
            
        Returns:
            Path to saved ensemble artifact directory
        """
        ensemble_path = self.artifacts_dir / f"ensemble_{ensemble_name}_{version}"
        ensemble_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save individual models
            models_dir = ensemble_path / "base_models"
            models_dir.mkdir(exist_ok=True)
            
            for model_name, model in models.items():
                model_calibrator = calibrators.get(model_name) if calibrators else None
                model_threshold = thresholds.get(model_name) if thresholds else None
                
                # Save each model as complete artifact
                self.save_complete_model(
                    model_name=model_name,
                    model=model,
                    calibrator=model_calibrator,
                    threshold=model_threshold,
                    feature_names=feature_info.get('feature_names') if feature_info else None,
                    categorical_features=feature_info.get('categorical_features') if feature_info else None,
                    version="ensemble_member"
                )
                
                # Create symlink in ensemble directory
                target_path = self.artifacts_dir / f"{model_name}_ensemble_member"
                link_path = models_dir / model_name
                if target_path.exists() and not link_path.exists():
                    try:
                        if os.name == 'nt':  # Windows
                            import shutil
                            shutil.copytree(target_path, link_path)
                        else:  # Unix/Linux
                            link_path.symlink_to(target_path)
                    except Exception:
                        # Fallback: just save path reference
                        with open(link_path.with_suffix('.txt'), 'w') as f:
                            f.write(str(target_path))
            
            # Save meta-model if present
            if meta_model is not None:
                meta_file = ensemble_path / "meta_model.joblib"
                joblib.dump(meta_model, meta_file)
                logger.info(f"✅ Saved meta-model: {meta_file}")
            
            # Save ensemble configuration
            ensemble_config = {
                'ensemble_name': ensemble_name,
                'version': version,
                'timestamp': datetime.now().isoformat(),
                'models': list(models.keys()),
                'ensemble_weights': ensemble_weights or {},
                'has_meta_model': meta_model is not None,
                'feature_info': feature_info or {},
                'deployment_ready': True
            }
            
            config_file = ensemble_path / "ensemble_config.json"
            with open(config_file, 'w') as f:
                json.dump(ensemble_config, f, indent=2, default=str)
            
            logger.info(f"🎯 Complete ensemble artifacts saved: {ensemble_path}")
            return str(ensemble_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save ensemble artifacts: {e}")
            raise
    
    def list_saved_models(self) -> List[Dict[str, str]]:
        """
        List all saved model artifacts.
        
        Returns:
            List of dictionaries with model info
        """
        models = []
        
        for path in self.artifacts_dir.iterdir():
            if path.is_dir():
                metadata_file = path / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        models.append({
                            'name': metadata.get('model_name', path.name),
                            'version': metadata.get('version', 'unknown'),
                            'type': metadata.get('model_type', 'unknown'),
                            'timestamp': metadata.get('timestamp', 'unknown'),
                            'path': str(path)
                        })
                    except Exception:
                        continue
        
        return sorted(models, key=lambda x: x['timestamp'], reverse=True)

def save_training_artifacts(models: Dict[str, Any],
                           calibrators: Optional[Dict[str, Any]] = None,
                           thresholds: Optional[Dict[str, float]] = None,
                           feature_info: Optional[Dict[str, Any]] = None,
                           ensemble_weights: Optional[Dict[str, float]] = None,
                           meta_model: Optional[Any] = None,
                           training_config: Optional[Any] = None,
                           performance_metrics: Optional[Dict[str, Any]] = None,
                           artifacts_dir: str = "models/artifacts") -> str:
    """
    Convenience function to save complete training artifacts.
    
    Args:
        models: Dictionary of trained models
        calibrators: Dictionary of calibrators
        thresholds: Dictionary of thresholds
        feature_info: Feature metadata
        ensemble_weights: Model weights
        meta_model: Meta-model
        training_config: Training configuration
        performance_metrics: Performance metrics
        artifacts_dir: Directory for artifacts
        
    Returns:
        Path to saved artifacts
    """
    manager = ModelArtifactManager(artifacts_dir)
    
    # Prepare metadata
    metadata = {
        'training_completed_at': datetime.now().isoformat(),
        'config': training_config.__dict__ if hasattr(training_config, '__dict__') else str(training_config),
        'performance_metrics': performance_metrics or {},
        'ensemble_weights': ensemble_weights or {}
    }
    
    # Save as ensemble
    return manager.save_ensemble_artifacts(
        ensemble_name="investment_committee",
        models=models,
        calibrators=calibrators,
        thresholds=thresholds,
        ensemble_weights=ensemble_weights,
        meta_model=meta_model,
        feature_info=feature_info,
        version=datetime.now().strftime("%Y%m%d_%H%M%S")
    )
