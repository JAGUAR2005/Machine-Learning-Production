# Final Model Leaderboard Report

## 1. Executive Leaderboard
| Rank | Algorithm | Price Prediction (R²) | Segment Accuracy | Verdict |
|------|-----------|------------------------|------------------|---------|
| 🏆 1 | **Random Forest** | **0.8713** | 81.1% | **Champion (Price)** |
| 🥈 2 | **XGBoost** | 0.8664 | **81.6%** | **Champion (Segments)** |
| 🥉 3 | **Decision Tree** | 0.8553 | 79.2% | Runner Up |
| 🏅 4 | **Logistic Regression**| 0.7519 | 73.7% | Baseline |
| 🏅 5 | **SVM (Linear)** | 0.7406 | 68.4% | Minimum Performance |

## 2. Key Insights
- **The Battle of Giants**: Random Forest and XGBoost are neck-and-neck. Random Forest is slightly better at exact price prediction, while XGBoost is superior at categorizing cars into market segments.
- **Complexity Matters**: Non-linear models (Forest, Boost, Trees) outperformed linear ones (SVM, Logistic) by a massive **12-15% margin**.
- **The Sweet Spot**: Decision Tree offers the best speed-to-performance ratio, making it great for quick iterations.

## 3. Deployment Recommendation
We recommend using **Random Forest** for the valuation engine due to its superior R² score, and **XGBoost** for the market classification engine.

*Report generated on: 2026-05-02*
