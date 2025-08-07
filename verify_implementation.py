#!/usr/bin/env python3
"""
Verify Current Model Implementation
===================================

Check the actual implementation in each model file to confirm balanced class weights.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_random_forest():
    """Check Random Forest implementation."""
    logger.info("Checking Random Forest implementation...")
    
    # Read the actual file content
    try:
        with open('models/random_forest_model.py', 'r') as f:
            content = f.read()
        
        # Check for class_weight in params
        if "'class_weight': 'balanced'" in content:
            logger.info("✅ Random Forest: class_weight='balanced' found in params")
        else:
            logger.warning("❌ Random Forest: class_weight='balanced' NOT found in params")
        
        # Check the model initialization
        import re
        model_init_pattern = r"self\.model = RandomForestClassifier\(\*\*self\.params\)"
        if re.search(model_init_pattern, content):
            logger.info("✅ Random Forest: Uses params in model initialization")
        else:
            logger.warning("❌ Random Forest: Doesn't use params in model initialization")
            
    except Exception as e:
        logger.error(f"Failed to check Random Forest: {e}")

def check_svm():
    """Check SVM implementation."""
    logger.info("Checking SVM implementation...")
    
    try:
        with open('models/svc_model.py', 'r') as f:
            content = f.read()
        
        # Check for class_weight in SVC constructor
        if "class_weight='balanced'" in content:
            logger.info("✅ SVM: class_weight='balanced' found in SVC constructor")
        else:
            logger.warning("❌ SVM: class_weight='balanced' NOT found in SVC constructor")
            
        # Find the SVC constructor line
        import re
        svc_pattern = r"SVC\([^)]*\)"
        matches = re.findall(svc_pattern, content, re.DOTALL)
        if matches:
            logger.info(f"SVC constructor found: {matches[0][:100]}...")
        
    except Exception as e:
        logger.error(f"Failed to check SVM: {e}")

def check_xgboost():
    """Check XGBoost implementation."""
    logger.info("Checking XGBoost implementation...")
    
    try:
        with open('models/xgboost_model.py', 'r') as f:
            content = f.read()
        
        # Check for scale_pos_weight calculation
        if "scale_pos_weight = n_neg / n_pos" in content:
            logger.info("✅ XGBoost: scale_pos_weight calculation found")
        else:
            logger.warning("❌ XGBoost: scale_pos_weight calculation NOT found")
        
        # Check for set_params call
        if "self.model.set_params(scale_pos_weight=" in content:
            logger.info("✅ XGBoost: set_params call found")
        else:
            logger.warning("❌ XGBoost: set_params call NOT found")
            
    except Exception as e:
        logger.error(f"Failed to check XGBoost: {e}")

def check_smote_ratio():
    """Check SMOTE ratio configuration."""
    logger.info("Checking SMOTE ratio configuration...")
    
    try:
        with open('config/training_config.py', 'r') as f:
            content = f.read()
        
        # Check for desired_ratio = 0.5
        if "desired_ratio: float = 0.5" in content:
            logger.info("✅ SMOTE: desired_ratio = 0.5 found in config")
        elif "desired_ratio: float = 0.6" in content:
            logger.warning("❌ SMOTE: desired_ratio = 0.6 (still 60/40)")
        else:
            logger.warning("❌ SMOTE: desired_ratio not found or has different value")
            
    except Exception as e:
        logger.error(f"Failed to check SMOTE config: {e}")

def main():
    """Run all checks."""
    logger.info("="*60)
    logger.info("VERIFYING CURRENT MODEL IMPLEMENTATIONS")
    logger.info("="*60)
    
    check_random_forest()
    logger.info("")
    
    check_svm()
    logger.info("")
    
    check_xgboost()
    logger.info("")
    
    check_smote_ratio()
    logger.info("")
    
    logger.info("="*60)
    logger.info("VERIFICATION COMPLETE")
    logger.info("="*60)

if __name__ == "__main__":
    main()
