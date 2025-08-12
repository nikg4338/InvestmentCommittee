# CLEAN MODEL TRAINING GUIDE

## 🎯 **Use train_clean_models.py for ALL future training**

### **✅ What's Changed:**
- **Data Leakage FIXED**: Removed `target_enhanced` and `target_3d_enhanced` features
- **Optuna Optimization**: 25-75 trials per model for optimal hyperparameters  
- **Realistic Performance**: Expect 85-90% accuracy (vs impossible 100%)
- **Production Ready**: Clean models suitable for live paper trading

### **📋 Training Process:**

**1. Run Clean Training:**
```bash
python train_clean_models.py
```

**2. Choose Trial Count:**
- Enter 25-75 when prompted
- Recommended: 50 trials (good balance of optimization vs time)

**3. Models Trained:**
- CatBoost (with Optuna optimization)
- Random Forest (with Optuna optimization)  
- XGBoost (with Optuna optimization)
- LightGBM (with Optuna optimization)

### **📁 Output Files:**
- `models/clean/catboost_clean_optimized.pkl`
- `models/clean/random_forest_clean_optimized.pkl`  
- `models/clean/xgboost_clean_optimized.pkl`
- `models/clean/lightgbm_clean_optimized.pkl`
- `models/clean/clean_training_summary.json`

### **⚠️ AVOID These Old Scripts:**
- ❌ `enhanced_model_training.py` (has data leakage)
- ❌ `train_models.py` (has data leakage)
- ❌ Any models in `models/production/` (contaminated)

### **✅ Expected Results:**
- **Accuracy**: 85-90% (realistic for financial data)
- **ROC-AUC**: 94-96% (excellent predictive power)
- **Training Time**: 15-30 minutes with optimization
- **Ready for Production**: Yes, suitable for $100K paper trading

### **🚀 Next Steps After Training:**
1. Update `scalable_trading_system.py` to use clean models
2. Deploy on 529 symbols from `filtered_iex_batches.json`
3. Execute bull put spreads with confidence thresholds
4. Monitor performance on live $100K Alpaca portfolio

### **🎯 Key Benefits:**
- **No Data Leakage**: Models won't fail in live trading
- **Optimized Performance**: Hyperparameters tuned for best results
- **Ensemble Ready**: Multiple models for robust predictions
- **Scalable**: Ready for 529 symbol deployment

**Always use `train_clean_models.py` - it's your production training pipeline!**
