# Machine Learning Model Evaluation Report
**Version**: 3.0 (Enhanced XGBoost with Feature Engineering)
**Trained**: 2026-05-02 18:01

## 1. Executive Summary
The model has been upgraded with **enhanced feature engineering** (log transforms, luxury indicators, mileage buckets, age-squared depreciation), **expanded hyperparameter search** (20 iterations, 5-fold CV for both models), and **stratified train-test splits**. Polynomial features were **removed** as XGBoost handles non-linearity natively.

## 2. Model Architecture
- **Target Transformation**: Log-Price prediction (log1p/expm1) to handle heteroscedasticity
- **Feature Set**: 13 features including 7 numeric, 4 categorical (OHE), 2 high-cardinality (Target Encoded)
- **Hyperparameter Tuning**: RandomizedSearchCV with 5-fold CV, 20 iterations (regression), 15 iterations (classification)
- **Stratified Split**: 80/20 train-test split stratified on price segments

## 3. Performance Metrics

### Regression (Price Estimation)
| Metric | Baseline (v2) | Current (v3) | Change |
|--------|---------------|--------------|--------|
| **R² Score** | 0.8643 | 0.8664 | +0.2% |
| **MAE** | $3,463 | $3,351.98 | - |
| **RMSE** | $6,525 | $6,459.03 | - |
| **MAPE** | N/A | 14.95% | - |

### Classification (Market Segmentation)
| Metric | Baseline (v2) | Current (v3) | Change |
|--------|---------------|--------------|--------|
| **Accuracy** | 81.8% | 81.6% | -0.2% |
| **Precision** | 81.9% | 81.7% | - |
| **Recall** | 81.8% | 81.6% | - |
| **F1 Score** | 81.8% | 81.7% | - |

## 4. Key Improvements
- Removed PolynomialFeatures (was causing noise for XGBoost)
- Added log_mileage, age_squared, is_luxury, mileage_bucket features
- Stratified train/test split for balanced classification
- Both models now have tuned hyperparameters via RandomizedSearchCV
- Expanded search space with regularization parameters (alpha, lambda, gamma)

## 5. Deployment Files
- **Regression Engine**: `xgb_pipeline.pkl`
- **Classification Engine**: `xgb_classifier_pipeline.pkl`
- **Training Metrics**: `training_metrics.json`
- **Residual Samples**: `sample_residuals.json`
- **Predicted vs Actual**: `predicted_vs_actual.json`
