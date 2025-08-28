#!/usr/bin/env python3
"""
Production Model Deployment System
================================

This module handles the deployment of trained models from staging to production,
including versioning, rollback capabilities, and production validation.
"""

import json
import shutil
import os
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class ProductionDeploymentSystem:
    """Manages model deployment from staging to production."""
    
    def __init__(self, reports_dir: str = 'reports', models_dir: str = 'models'):
        """Initialize production deployment system."""
        self.reports_dir = Path(reports_dir)
        self.models_dir = Path(models_dir)
        self.production_dir = self.models_dir / 'production'
        self.staging_dir = self.models_dir / 'staging'
        self.archive_dir = self.models_dir / 'archive'
        
        # Create directories
        for directory in [self.production_dir, self.staging_dir, self.archive_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def evaluate_batch_for_production(self, batch_num: int) -> Dict[str, Any]:
        """Evaluate a batch for production readiness."""
        try:
            batch_dir = self.reports_dir / f'batch_{batch_num}'
            if not batch_dir.exists():
                return {"eligible": False, "reason": "Batch directory not found"}
            
            # Load telemetry data
            telemetry_file = batch_dir / f'batch_{batch_num}_telemetry.json'
            if not telemetry_file.exists():
                return {"eligible": False, "reason": "Telemetry data not found"}
            
            with open(telemetry_file, 'r') as f:
                telemetry = json.load(f)
            
            # Production readiness criteria
            criteria = {
                "min_pr_auc": 0.05,
                "min_models_trained": 3,
                "gate_passed": True,
                "training_successful": True,
                "min_samples": 1000
            }
            
            # Evaluate criteria
            evaluation = {
                "batch": batch_num,
                "pr_auc": telemetry.get('pr_auc_meta', 0.0),
                "models_trained": telemetry.get('models_trained', 0),
                "gate": telemetry.get('gate', 'FAILED'),
                "training_successful": telemetry.get('training_successful', False),
                "samples_processed": telemetry.get('samples_processed', 0),
                "eligible": True,
                "reasons": []
            }
            
            # Check each criterion
            if evaluation["pr_auc"] < criteria["min_pr_auc"]:
                evaluation["eligible"] = False
                evaluation["reasons"].append(f"PR-AUC too low: {evaluation['pr_auc']:.4f} < {criteria['min_pr_auc']}")
            
            if evaluation["models_trained"] < criteria["min_models_trained"]:
                evaluation["eligible"] = False
                evaluation["reasons"].append(f"Insufficient models: {evaluation['models_trained']} < {criteria['min_models_trained']}")
            
            if evaluation["gate"] != "PASSED":
                evaluation["eligible"] = False
                evaluation["reasons"].append(f"Failed quality gate: {evaluation['gate']}")
            
            if not evaluation["training_successful"]:
                evaluation["eligible"] = False
                evaluation["reasons"].append("Training was not successful")
            
            if evaluation["samples_processed"] < criteria["min_samples"]:
                evaluation["eligible"] = False
                evaluation["reasons"].append(f"Insufficient samples: {evaluation['samples_processed']} < {criteria['min_samples']}")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Failed to evaluate batch {batch_num}: {e}")
            return {"eligible": False, "reason": f"Evaluation error: {e}"}
    
    def find_best_production_candidate(self) -> Optional[Dict[str, Any]]:
        """Find the best batch for production deployment."""
        try:
            if not self.reports_dir.exists():
                logger.error("Reports directory not found")
                return None
            
            # Get all batch directories
            batch_dirs = [d for d in self.reports_dir.glob('batch_*') if d.is_dir()]
            if not batch_dirs:
                logger.error("No batch directories found")
                return None
            
            candidates = []
            
            # Evaluate each batch
            for batch_dir in batch_dirs:
                try:
                    batch_num = int(batch_dir.name.replace('batch_', ''))
                    evaluation = self.evaluate_batch_for_production(batch_num)
                    
                    if evaluation.get("eligible", False):
                        candidates.append(evaluation)
                        logger.info(f"✅ Batch {batch_num} is production eligible (PR-AUC: {evaluation['pr_auc']:.4f})")
                    else:
                        reasons = evaluation.get("reasons", [evaluation.get("reason", "Unknown")])
                        logger.info(f"❌ Batch {batch_num} not eligible: {'; '.join(reasons)}")
                except ValueError:
                    logger.warning(f"Invalid batch directory name: {batch_dir.name}")
                    continue
                except Exception as e:
                    logger.warning(f"Failed to evaluate {batch_dir.name}: {e}")
                    continue
            
            if not candidates:
                logger.warning("No eligible batches found for production")
                return None
            
            # Select best candidate (highest PR-AUC)
            best_candidate = max(candidates, key=lambda x: x['pr_auc'])
            logger.info(f"🏆 Best production candidate: Batch {best_candidate['batch']} (PR-AUC: {best_candidate['pr_auc']:.4f})")
            
            return best_candidate
            
        except Exception as e:
            logger.error(f"Failed to find best production candidate: {e}")
            return None
    
    def copy_models_to_staging(self, batch_num: int) -> bool:
        """Copy models from batch results to staging area."""
        try:
            batch_dir = self.reports_dir / f'batch_{batch_num}'
            results_dir = batch_dir / 'results'
            
            if not results_dir.exists():
                logger.error(f"Results directory not found for batch {batch_num}")
                return False
            
            # Clear staging directory
            if self.staging_dir.exists():
                shutil.rmtree(self.staging_dir)
            self.staging_dir.mkdir(parents=True)
            
            # Copy model files
            model_files = list(results_dir.glob('*.pkl')) + list(results_dir.glob('*.json'))
            copied_files = []
            
            for model_file in model_files:
                dest_file = self.staging_dir / model_file.name
                shutil.copy2(model_file, dest_file)
                copied_files.append(model_file.name)
                logger.debug(f"Copied {model_file.name} to staging")
            
            if copied_files:
                logger.info(f"✅ Copied {len(copied_files)} model files to staging: {', '.join(copied_files)}")
                return True
            else:
                logger.warning("No model files found to copy")
                return False
                
        except Exception as e:
            logger.error(f"Failed to copy models to staging for batch {batch_num}: {e}")
            return False
    
    def validate_staged_models(self, batch_info: Dict[str, Any]) -> bool:
        """Validate models in staging area before production deployment."""
        try:
            if not self.staging_dir.exists() or not any(self.staging_dir.iterdir()):
                logger.error("Staging directory is empty")
                return False
            
            # Check for required model files
            required_extensions = ['.pkl', '.json']
            found_files = []
            
            for ext in required_extensions:
                files = list(self.staging_dir.glob(f'*{ext}'))
                found_files.extend(files)
            
            if not found_files:
                logger.error("No valid model files found in staging")
                return False
            
            # Validate pickle files can be loaded
            pkl_files = list(self.staging_dir.glob('*.pkl'))
            valid_models = 0
            
            for pkl_file in pkl_files:
                try:
                    with open(pkl_file, 'rb') as f:
                        model = pickle.load(f)
                    
                    # Basic validation - check if it's a trained model
                    if hasattr(model, 'predict') or hasattr(model, 'predict_proba'):
                        valid_models += 1
                        logger.debug(f"✅ Validated model: {pkl_file.name}")
                    else:
                        logger.warning(f"⚠️ Invalid model format: {pkl_file.name}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load model {pkl_file.name}: {e}")
            
            # Validation criteria
            min_valid_models = 3
            if valid_models >= min_valid_models:
                logger.info(f"✅ Staging validation passed: {valid_models} valid models found")
                return True
            else:
                logger.error(f"❌ Staging validation failed: only {valid_models} valid models (minimum {min_valid_models})")
                return False
                
        except Exception as e:
            logger.error(f"Failed to validate staged models: {e}")
            return False
    
    def create_production_version(self, batch_info: Dict[str, Any]) -> str:
        """Create a new production version from staged models."""
        try:
            # Generate version identifier
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            version_id = f"v{timestamp}_batch{batch_info['batch']}"
            version_dir = self.production_dir / version_id
            
            # Create version directory
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy models from staging to version directory
            if not self.staging_dir.exists():
                raise ValueError("Staging directory not found")
            
            copied_files = []
            for file_path in self.staging_dir.iterdir():
                if file_path.is_file():
                    dest_path = version_dir / file_path.name
                    shutil.copy2(file_path, dest_path)
                    copied_files.append(file_path.name)
            
            # Create production manifest
            manifest = {
                "version_id": version_id,
                "batch_source": batch_info['batch'],
                "deployment_timestamp": datetime.now().isoformat(),
                "pr_auc": batch_info['pr_auc'],
                "models_included": copied_files,
                "performance_metrics": {
                    "pr_auc": batch_info['pr_auc'],
                    "gate_status": batch_info['gate'],
                    "models_trained": batch_info['models_trained'],
                    "samples_processed": batch_info['samples_processed']
                },
                "validation_status": "PASSED",
                "deployment_notes": f"Deployed from batch {batch_info['batch']} with PR-AUC {batch_info['pr_auc']:.4f}"
            }
            
            # Save manifest
            manifest_path = version_dir / 'production_manifest.json'
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"✅ Created production version: {version_id}")
            logger.info(f"📁 Location: {version_dir}")
            logger.info(f"📊 Models: {len(copied_files)} files")
            
            return version_id
            
        except Exception as e:
            logger.error(f"Failed to create production version: {e}")
            return ""
    
    def update_production_symlink(self, version_id: str) -> bool:
        """Update the 'latest' symlink to point to the new version."""
        try:
            version_dir = self.production_dir / version_id
            if not version_dir.exists():
                logger.error(f"Version directory not found: {version_id}")
                return False
            
            latest_link = self.production_dir / 'latest'
            
            # Remove existing symlink
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            
            # Create new symlink
            try:
                # Use relative path for better portability
                relative_path = Path(version_id)
                os.symlink(relative_path, latest_link, target_is_directory=True)
                logger.info(f"✅ Updated 'latest' symlink to {version_id}")
                return True
            except OSError:
                # Fallback: copy directory if symlinks not supported
                if latest_link.exists():
                    shutil.rmtree(latest_link)
                shutil.copytree(version_dir, latest_link)
                logger.info(f"✅ Copied {version_id} to 'latest' directory (symlink not supported)")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update production symlink: {e}")
            return False
    
    def archive_previous_version(self) -> bool:
        """Archive the previous production version."""
        try:
            # Find current production versions (excluding 'latest')
            prod_versions = [d for d in self.production_dir.iterdir() 
                           if d.is_dir() and d.name != 'latest' and d.name.startswith('v')]
            
            if len(prod_versions) <= 1:
                logger.info("No previous versions to archive")
                return True
            
            # Sort by creation time, archive older versions (keep last 3)
            prod_versions.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            versions_to_archive = prod_versions[3:]  # Keep 3 most recent
            
            for version_dir in versions_to_archive:
                archive_dest = self.archive_dir / version_dir.name
                
                # Move to archive
                if archive_dest.exists():
                    shutil.rmtree(archive_dest)
                shutil.move(str(version_dir), str(archive_dest))
                
                logger.info(f"📦 Archived version: {version_dir.name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to archive previous versions: {e}")
            return False
    
    def deploy_to_production(self, batch_num: Optional[int] = None) -> Dict[str, Any]:
        """Complete deployment process from staging to production."""
        try:
            logger.info("🚀 Starting production deployment process...")
            
            # Find best candidate if not specified
            if batch_num is None:
                best_candidate = self.find_best_production_candidate()
                if not best_candidate:
                    return {"success": False, "error": "No eligible batches found for production"}
                batch_info = best_candidate
                batch_num = batch_info['batch']
            else:
                # Evaluate specified batch
                batch_info = self.evaluate_batch_for_production(batch_num)
                if not batch_info.get("eligible", False):
                    reasons = batch_info.get("reasons", [batch_info.get("reason", "Unknown")])
                    return {"success": False, "error": f"Batch {batch_num} not eligible: {'; '.join(reasons)}"}
            
            logger.info(f"📊 Deploying batch {batch_num} with PR-AUC: {batch_info['pr_auc']:.4f}")
            
            # Step 1: Copy models to staging
            if not self.copy_models_to_staging(batch_num):
                return {"success": False, "error": "Failed to copy models to staging"}
            
            # Step 2: Validate staged models
            if not self.validate_staged_models(batch_info):
                return {"success": False, "error": "Staged model validation failed"}
            
            # Step 3: Create production version
            version_id = self.create_production_version(batch_info)
            if not version_id:
                return {"success": False, "error": "Failed to create production version"}
            
            # Step 4: Update production symlink
            if not self.update_production_symlink(version_id):
                return {"success": False, "error": "Failed to update production symlink"}
            
            # Step 5: Archive old versions
            self.archive_previous_version()
            
            # Create deployment summary
            deployment_summary = {
                "success": True,
                "version_id": version_id,
                "batch_source": batch_num,
                "pr_auc": batch_info['pr_auc'],
                "deployment_time": datetime.now().isoformat(),
                "models_deployed": len(list((self.production_dir / version_id).glob('*.pkl'))),
                "production_path": str(self.production_dir / 'latest')
            }
            
            # Save deployment log
            deployment_log_path = self.production_dir / 'deployment_history.json'
            deployment_history = []
            
            if deployment_log_path.exists():
                try:
                    with open(deployment_log_path, 'r') as f:
                        deployment_history = json.load(f)
                except:
                    deployment_history = []
            
            deployment_history.append(deployment_summary)
            
            # Keep only last 10 deployments
            deployment_history = deployment_history[-10:]
            
            with open(deployment_log_path, 'w') as f:
                json.dump(deployment_history, f, indent=2)
            
            logger.info("🎉 Production deployment completed successfully!")
            logger.info(f"✅ Version: {version_id}")
            logger.info(f"📁 Path: {self.production_dir / 'latest'}")
            logger.info(f"📊 Performance: PR-AUC {batch_info['pr_auc']:.4f}")
            
            return deployment_summary
            
        except Exception as e:
            logger.error(f"Production deployment failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_production_status(self) -> Dict[str, Any]:
        """Get current production deployment status."""
        try:
            latest_path = self.production_dir / 'latest'
            
            if not latest_path.exists():
                return {"status": "NO_DEPLOYMENT", "message": "No models deployed to production"}
            
            # Load manifest
            manifest_path = latest_path / 'production_manifest.json'
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                
                return {
                    "status": "DEPLOYED",
                    "version_id": manifest.get('version_id', 'unknown'),
                    "batch_source": manifest.get('batch_source', 'unknown'),
                    "deployment_time": manifest.get('deployment_timestamp', 'unknown'),
                    "pr_auc": manifest.get('pr_auc', 0.0),
                    "models_count": len(manifest.get('models_included', [])),
                    "path": str(latest_path)
                }
            else:
                return {"status": "DEPLOYED_NO_MANIFEST", "path": str(latest_path)}
                
        except Exception as e:
            logger.error(f"Failed to get production status: {e}")
            return {"status": "ERROR", "error": str(e)}
    
    def rollback_production(self, target_version: Optional[str] = None) -> Dict[str, Any]:
        """Rollback production to a previous version."""
        try:
            if target_version is None:
                # Find previous version
                prod_versions = [d for d in self.production_dir.iterdir() 
                               if d.is_dir() and d.name != 'latest' and d.name.startswith('v')]
                
                if len(prod_versions) < 2:
                    return {"success": False, "error": "No previous version available for rollback"}
                
                # Sort by creation time and get second most recent
                prod_versions.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                target_version = prod_versions[1].name
            
            target_path = self.production_dir / target_version
            if not target_path.exists():
                return {"success": False, "error": f"Target version not found: {target_version}"}
            
            # Update symlink to target version
            if self.update_production_symlink(target_version):
                logger.info(f"✅ Production rolled back to version: {target_version}")
                return {
                    "success": True,
                    "version_id": target_version,
                    "rollback_time": datetime.now().isoformat()
                }
            else:
                return {"success": False, "error": "Failed to update symlink during rollback"}
                
        except Exception as e:
            logger.error(f"Production rollback failed: {e}")
            return {"success": False, "error": str(e)}


# Global deployment system instance
production_deployer = ProductionDeploymentSystem()
