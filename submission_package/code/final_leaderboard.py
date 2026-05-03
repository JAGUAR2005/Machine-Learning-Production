import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Results data
results = [
    {"Algorithm": "XGBoost", "R2": 0.8682, "Accuracy": 0.8171, "Color": "red"},
    {"Algorithm": "Random Forest", "R2": 0.8713, "Accuracy": 0.8115, "Color": "green"},
    {"Algorithm": "Decision Tree", "R2": 0.8553, "Accuracy": 0.7916, "Color": "orange"},
    {"Algorithm": "Logistic Reg", "R2": 0.7519, "Accuracy": 0.7370, "Color": "blue"},
    {"Algorithm": "SVM", "R2": 0.7406, "Accuracy": 0.6841, "Color": "purple"}
]

df = pd.DataFrame(results).sort_values(by="R2", ascending=False)

# Create Comparison Plot
fig, ax1 = plt.subplots(figsize=(12, 7))

x = np.arange(len(df))
width = 0.35

rects1 = ax1.bar(x - width/2, df['R2'], width, label='R² Score (Price)', color='skyblue', alpha=0.8)
ax1.set_ylabel('R² Score')
ax1.set_title('Final Algorithm Comparison: Resale Intelligence Systems')
ax1.set_xticks(x)
ax1.set_xticklabels(df['Algorithm'])
ax1.set_ylim(0.6, 0.95)

ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, df['Accuracy'], width, label='Accuracy (Segments)', color='salmon', alpha=0.8)
ax2.set_ylabel('Accuracy Score')
ax2.set_ylim(0.6, 0.95)

# Add value labels
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

autolabel(rects1, ax1)
autolabel(rects2, ax2)

fig.tight_layout()
plt.legend(handles=[rects1, rects2], loc='upper right')
plt.savefig('plots/final_leaderboard_comparison.png', dpi=150)
plt.close()

# Generate Markdown Summary
summary = f"""# Final Model Leaderboard Report

## 1. Executive Leaderboard
| Rank | Algorithm | Price Prediction (R²) | Segment Accuracy | Verdict |
|------|-----------|------------------------|------------------|---------|
| 1 | **Random Forest** | **0.8713** | 81.1% | **Champion (Price)** |
| 2 | **XGBoost** | 0.8682 | **81.7%** | **Champion (Segments)** |
| 3 | Decision Tree | 0.8553 | 79.2% | Runner Up |
| 4 | Logistic Reg | 0.7519 | 73.7% | Baseline |
| 5 | SVM | 0.7406 | 68.4% | Minimum Performance |

## 2. Key Insights
- **The Battle of Giants**: Random Forest and XGBoost are neck-and-neck. Random Forest is slightly better at exact price prediction, while XGBoost is superior at categorizing cars into market segments.
- **Complexity Matters**: Non-linear models (Forest, Boost, Trees) outperformed linear ones (SVM, Logistic) by a massive **12-15% margin**.
- **The Sweet Spot**: Decision Tree offers the best speed-to-performance ratio, making it great for quick iterations.

## 3. Deployment Recommendation
We recommend using **Random Forest** for the valuation engine due to its superior R² score, and **XGBoost** for the market classification engine.

*Report generated on: {pd.Timestamp.now().strftime('%Y-%m-%d')}*
"""

with open('FINAL_COMPARISON_REPORT.md', 'w') as f:
    f.write(summary)

print("  Final leaderboard and report generated.")
