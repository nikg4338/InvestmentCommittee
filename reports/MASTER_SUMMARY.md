# Investment Committee - Master Training Summary

**Generated:** 2025-08-02 23:01:38
**Total Processing Time:** 188.5 seconds (3.1 minutes)

## Batch Processing Results

### Successful Batches (11)
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11

### Failed Batches (0)
None

## Configuration Used
- **Training Config:** extreme_imbalance (optimized for financial data)
- **Target Column:** target (buy/sell signals)
- **Visualization:** All plots saved to reports/batch_X/plots/
- **Metrics Export:** CSV files in reports/batch_X/results/
- **Logging Level:** INFO (detailed progress tracking)

## Next Steps
1. **Review Individual Batches:** Check `reports/batch_X/BATCH_SUMMARY.md` for each batch
2. **Compare Performance:** Use `reports/batch_X/results/performance_summary.csv` files
3. **Analyze Plots:** Visual analysis in `reports/batch_X/plots/` folders
4. **Aggregate Results:** Consider combining successful batches for meta-analysis

## Directory Structure
```
reports/
├── MASTER_SUMMARY.md          # This file
├── batch_1/                   # Batch 1 results
├── batch_2/                   # Batch 2 results
├── ...
└── batch_N/                   # Batch N results
```

## Model Performance Overview
Each batch was trained with the Committee of Five ensemble:
- XGBoost
- LightGBM  
- CatBoost
- Random Forest
- Support Vector Machine
- Meta-model (LogisticRegression)
- Final Ensemble (rank-and-vote)

For detailed performance metrics, see individual batch result files.
