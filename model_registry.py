"""
Model Registry for Investment Committee Training Pipeline
Centralized tracking of model versions, performance, and deployment status.
"""

import json
import os
import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Centralized registry for tracking model lifecycle, performance, and deployment.
    Provides version control, performance comparison, and deployment management.
    """
    
    def __init__(self, registry_path: str = "models/registry"):
        """Initialize the model registry."""
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Registry file paths
        self.metadata_file = self.registry_path / "model_metadata.json"
        self.performance_file = self.registry_path / "performance_history.json"
        self.deployment_file = self.registry_path / "deployment_status.json"
        
        # Initialize registry files if they don't exist
        self._initialize_registry_files()
        
    def _initialize_registry_files(self):
        """Initialize registry files with empty structures."""
        if not self.metadata_file.exists():
            with open(self.metadata_file, 'w') as f:
                json.dump({}, f, indent=2)
                
        if not self.performance_file.exists():
            with open(self.performance_file, 'w') as f:
                json.dump({}, f, indent=2)
                
        if not self.deployment_file.exists():
            with open(self.deployment_file, 'w') as f:
                json.dump({
                    "production": {},
                    "staging": {},
                    "clean": {},
                    "development": {}
                }, f, indent=2)
    
    def register_model(self, 
                      model_id: str,
                      model_type: str,
                      batch_name: str,
                      model_path: str,
                      hyperparameters: Dict[str, Any],
                      performance_metrics: Dict[str, float],
                      features: List[str],
                      training_config: Dict[str, Any],
                      model_size_mb: Optional[float] = None) -> str:
        """
        Register a new model in the registry.
        
        Args:
            model_id: Unique identifier for the model
            model_type: Type of model (xgboost, lightgbm, etc.)
            batch_name: Name of the training batch
            model_path: Path to the saved model file
            hyperparameters: Model hyperparameters used
            performance_metrics: Model performance scores
            features: List of feature names used
            training_config: Configuration used during training
            model_size_mb: Size of model file in MB
            
        Returns:
            Version string for the registered model
        """
        try:
            # Load existing metadata
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Generate version number
            existing_versions = [
                v['version'] for v in metadata.values() 
                if v.get('model_type') == model_type and v.get('batch_name') == batch_name
            ]
            version = f"v{len(existing_versions) + 1:03d}"
            full_model_id = f"{model_id}_{version}"
            
            # Calculate model size if not provided
            if model_size_mb is None and os.path.exists(model_path):
                model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            
            # Create model metadata
            model_metadata = {
                "model_id": model_id,
                "full_model_id": full_model_id,
                "version": version,
                "model_type": model_type,
                "batch_name": batch_name,
                "model_path": model_path,
                "hyperparameters": hyperparameters,
                "features": features,
                "feature_count": len(features),
                "training_config": training_config,
                "model_size_mb": model_size_mb,
                "registration_time": datetime.datetime.now().isoformat(),
                "status": "registered"
            }
            
            # Add to metadata
            metadata[full_model_id] = model_metadata
            
            # Save updated metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Register performance metrics
            self.register_performance(full_model_id, performance_metrics)
            
            logger.info(f"✅ Model registered: {full_model_id}")
            return full_model_id
            
        except Exception as e:
            logger.error(f"Failed to register model {model_id}: {e}")
            return None
    
    def register_performance(self, model_id: str, metrics: Dict[str, float]):
        """Register performance metrics for a model."""
        try:
            # Load existing performance data
            with open(self.performance_file, 'r') as f:
                performance = json.load(f)
            
            # Add timestamp to metrics
            timestamped_metrics = {
                **metrics,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Initialize model performance history if needed
            if model_id not in performance:
                performance[model_id] = []
            
            # Add new performance record
            performance[model_id].append(timestamped_metrics)
            
            # Save updated performance data
            with open(self.performance_file, 'w') as f:
                json.dump(performance, f, indent=2, default=str)
            
            logger.info(f"✅ Performance registered for {model_id}")
            
        except Exception as e:
            logger.error(f"Failed to register performance for {model_id}: {e}")
    
    def promote_model(self, model_id: str, target_environment: str) -> bool:
        """
        Promote a model to a different environment.
        
        Args:
            model_id: Full model ID
            target_environment: Target environment (production, staging, clean, development)
            
        Returns:
            Success status
        """
        try:
            # Load deployment status
            with open(self.deployment_file, 'r') as f:
                deployment = json.load(f)
            
            # Load model metadata
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            if model_id not in metadata:
                logger.error(f"Model {model_id} not found in registry")
                return False
            
            # Update deployment status
            if target_environment not in deployment:
                deployment[target_environment] = {}
            
            deployment[target_environment][model_id] = {
                "promoted_at": datetime.datetime.now().isoformat(),
                "model_type": metadata[model_id]["model_type"],
                "batch_name": metadata[model_id]["batch_name"],
                "version": metadata[model_id]["version"]
            }
            
            # Update model status
            metadata[model_id]["status"] = f"deployed_{target_environment}"
            metadata[model_id]["last_deployment"] = {
                "environment": target_environment,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Save updates
            with open(self.deployment_file, 'w') as f:
                json.dump(deployment, f, indent=2, default=str)
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"✅ Model {model_id} promoted to {target_environment}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote model {model_id}: {e}")
            return False
    
    def get_best_models(self, 
                       metric: str = "pr_auc", 
                       top_k: int = 5,
                       model_type: Optional[str] = None,
                       batch_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get the best performing models based on a specific metric.
        
        Args:
            metric: Performance metric to rank by
            top_k: Number of top models to return
            model_type: Filter by specific model type
            batch_name: Filter by specific batch
            
        Returns:
            List of best model information
        """
        try:
            # Load data
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            with open(self.performance_file, 'r') as f:
                performance = json.load(f)
            
            # Collect model performance data
            model_scores = []
            for model_id, model_info in metadata.items():
                # Apply filters
                if model_type and model_info.get("model_type") != model_type:
                    continue
                if batch_name and model_info.get("batch_name") != batch_name:
                    continue
                
                # Get latest performance metrics
                if model_id in performance and performance[model_id]:
                    latest_metrics = performance[model_id][-1]
                    if metric in latest_metrics:
                        model_scores.append({
                            "model_id": model_id,
                            "model_type": model_info["model_type"],
                            "batch_name": model_info["batch_name"],
                            "version": model_info["version"],
                            "score": latest_metrics[metric],
                            "all_metrics": latest_metrics,
                            "metadata": model_info
                        })
            
            # Sort by score (descending) and return top k
            model_scores.sort(key=lambda x: x["score"], reverse=True)
            return model_scores[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to get best models: {e}")
            return []
    
    def get_model_history(self, model_id: str) -> Dict[str, Any]:
        """Get complete history and information for a specific model."""
        try:
            # Load all data
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            with open(self.performance_file, 'r') as f:
                performance = json.load(f)
            
            with open(self.deployment_file, 'r') as f:
                deployment = json.load(f)
            
            if model_id not in metadata:
                return {}
            
            # Compile complete history
            history = {
                "metadata": metadata[model_id],
                "performance_history": performance.get(model_id, []),
                "deployment_history": []
            }
            
            # Find deployment history
            for env, models in deployment.items():
                if model_id in models:
                    history["deployment_history"].append({
                        "environment": env,
                        **models[model_id]
                    })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get model history for {model_id}: {e}")
            return {}
    
    def cleanup_old_models(self, 
                          keep_versions: int = 3,
                          keep_days: int = 30) -> Dict[str, int]:
        """
        Clean up old model versions to save space.
        
        Args:
            keep_versions: Number of recent versions to keep per model type/batch
            keep_days: Number of days of models to keep regardless of version count
            
        Returns:
            Cleanup statistics
        """
        try:
            # Load metadata
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=keep_days)
            cleanup_stats = {"models_removed": 0, "space_freed_mb": 0}
            
            # Group models by type and batch
            model_groups = {}
            for model_id, model_info in metadata.items():
                key = f"{model_info['model_type']}_{model_info['batch_name']}"
                if key not in model_groups:
                    model_groups[key] = []
                model_groups[key].append((model_id, model_info))
            
            # Process each group
            for group_models in model_groups.values():
                # Sort by registration time (newest first)
                group_models.sort(
                    key=lambda x: datetime.datetime.fromisoformat(x[1]['registration_time']), 
                    reverse=True
                )
                
                # Mark models for deletion
                models_to_remove = []
                for i, (model_id, model_info) in enumerate(group_models):
                    registration_time = datetime.datetime.fromisoformat(model_info['registration_time'])
                    
                    # Keep if within version limit or recent enough
                    if i >= keep_versions and registration_time < cutoff_date:
                        # Don't remove deployed models
                        if model_info.get('status', '').startswith('deployed_'):
                            continue
                        models_to_remove.append((model_id, model_info))
                
                # Remove marked models
                for model_id, model_info in models_to_remove:
                    try:
                        # Remove model file if it exists
                        model_path = model_info.get('model_path')
                        if model_path and os.path.exists(model_path):
                            file_size = os.path.getsize(model_path) / (1024 * 1024)
                            os.remove(model_path)
                            cleanup_stats["space_freed_mb"] += file_size
                        
                        # Remove from metadata
                        del metadata[model_id]
                        cleanup_stats["models_removed"] += 1
                        
                        logger.info(f"🗑️ Removed old model: {model_id}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to remove model {model_id}: {e}")
            
            # Save updated metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"✅ Cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Failed to cleanup old models: {e}")
            return {"models_removed": 0, "space_freed_mb": 0}
    
    def generate_registry_report(self) -> str:
        """Generate a comprehensive registry status report."""
        try:
            # Load all data
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            with open(self.performance_file, 'r') as f:
                performance = json.load(f)
            
            with open(self.deployment_file, 'r') as f:
                deployment = json.load(f)
            
            # Generate report
            report = []
            report.append("=" * 80)
            report.append("MODEL REGISTRY REPORT")
            report.append("=" * 80)
            
            # Summary statistics
            total_models = len(metadata)
            model_types = set(m['model_type'] for m in metadata.values())
            batches = set(m['batch_name'] for m in metadata.values())
            total_size = sum(m.get('model_size_mb', 0) or 0 for m in metadata.values())
            
            report.append(f"Total Models: {total_models}")
            report.append(f"Model Types: {', '.join(sorted(model_types))}")
            report.append(f"Batches: {', '.join(sorted(batches))}")
            report.append(f"Total Storage: {total_size:.1f} MB")
            report.append("")
            
            # Deployment summary
            report.append("DEPLOYMENT STATUS:")
            for env, models in deployment.items():
                report.append(f"  {env.upper()}: {len(models)} models")
            report.append("")
            
            # Top performers
            best_models = self.get_best_models(metric="pr_auc", top_k=5)
            if best_models:
                report.append("TOP 5 MODELS (by PR-AUC):")
                for i, model in enumerate(best_models, 1):
                    report.append(f"  {i}. {model['model_id']} ({model['model_type']}) - {model['score']:.4f}")
                report.append("")
            
            # Recent activity
            recent_models = sorted(
                metadata.items(),
                key=lambda x: x[1]['registration_time'],
                reverse=True
            )[:5]
            
            if recent_models:
                report.append("RECENT REGISTRATIONS:")
                for model_id, model_info in recent_models:
                    reg_time = datetime.datetime.fromisoformat(model_info['registration_time'])
                    report.append(f"  {model_id} - {reg_time.strftime('%Y-%m-%d %H:%M')}")
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Failed to generate registry report: {e}")
            return "Failed to generate registry report"
