# Communities & Crime – Regression with XGBoost + Interpretability (SHAP)

## Project overview
This project builds regression models to predict community-level violent crime rate (`ViolentCrimesPerPop`) using the UCI Communities and Crime dataset.  
The workflow includes data preprocessing, model training (with a focus on XGBoost), evaluation on a held-out test set, and interpretability analyses using global and local feature importance (including SHAP).

## Repository structure
- `notebooks/`  
  Jupyter notebooks for EDA, model training, evaluation, and plots.
- `src/`  
  Python scripts/functions used by notebooks (model training, utility functions, plotting).
- `results/`  
  Saved outputs (predictions, trained models, figures).  
  Examples:
  - `results/xgb_test_predictions.csv`
  - `results/xgb_best_model.json`
  - `results/shap_global_top10.png`
  - `results/local_shap_sample_66.png`

## Key results
- Metric(s): RMSE and R² on the test set.
- Feature importance:
  - XGBoost built-in importance (with variability across random seeds).
  - SHAP global importance (mean |SHAP|).
  - SHAP local explanation for selected samples.

## Reproducibility
### Python and package versions
Developed and tested with:
- Python: **3.12.x**
- Key packages (exact versions listed in `environment.yml`)

### Environment setup (recommended: conda)
1. Create the environment:
   ```bash
   conda env create -f environment.yml
