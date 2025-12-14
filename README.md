# Communities & Crime – Regression with XGBoost + Interpretability (SHAP)

## Project overview
This project builds regression models to predict community-level violent crime rate (`ViolentCrimesPerPop`) using the UCI Communities and Crime dataset.  
The workflow includes data preprocessing, model training (with a focus on XGBoost and Reduced-feature models), evaluation on a held-out test set, and interpretability analyses using global and local feature importance (including SHAP).

## Repository structure
- `figures/`  
  figures for EDA, model training, evaluation, feature importances and plots.
  - `figures/shap_global_top10.png`
  - `figures/local_shap_sample_66.png`
- `src/`  
  Python scripts/functions used by notebooks (model training, utility functions, plotting).
- `results/`  
  Saved outputs (predictions, trained models, figures).  
  Examples:
  - `results/xgb_test_predictions.csv`
  - `results/xgb_best_model.json`


## Key results
- Metric(s): RMSE and R² on the test set.
- Missing value handling: reduces-feature-models, Xgboost
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
