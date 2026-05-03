# Global Car Resale Intelligence System (Ensemble Edition)

An advanced Machine Learning platform designed to predict vehicle resale values across global markets (India, Europe, and Asia/UK) using a multi-algorithm ensemble (XGBoost, Random Forest, SVM, Decision Tree, and Logistic Regression).

## 🚀 Key Features

- **Ensemble Neural Engine**: Integrates 5 top-tier algorithms to capture both linear and non-linear market volatility.
- **Model Inspector**: Interactive deep-dive module allowing technical audit of each algorithm's unique output (e.g., SVM Hyperplanes, Decision Tree maps).
- **Multi-Market Capability**: Integrated data pipelines for India, Europe, and the UK with automated brand-model filtering.
- **Dynamic Multi-Currency**: Real-time conversion support for USD ($), INR (₹), EUR (€), and GBP (£) powered by live exchange rates.
- **High-Fidelity Dashboard**: A professional Glassmorphism interface with **Spider Matrix** validation and **Shadowy Travelling Glow** animations.

## 📊 Performance Leaderboard

The system identifies the optimal model for specific tasks based on audited metrics:

| Rank | Algorithm | Task Excellence | Performance (R²) | Accuracy |
|------|-----------|-----------------|------------------|----------|
| 🏆 1 | **Random Forest** | Price Prediction | **0.8713** | 81.1% |
| 🥈 2 | **XGBoost** | Market Segmenting | 0.8664 | **81.6%** |
| 🥉 3 | **Decision Tree** | Interpretability | 0.8553 | 79.2% |
| 🏅 4 | **Logistic Regression**| Baseline Accuracy | 0.7519 | 73.7% |
| 🏅 5 | **SVM (Linear)** | Vector Stability | 0.7406 | 68.4% |

## 🛠️ Tech Stack

- **Backend**: Python 3.9, FastAPI, Uvicorn
- **Machine Learning**: XGBoost, Scikit-learn, Pandas, Joblib
- **Frontend**: React 19, TypeScript, Vite, Framer Motion, Recharts, Lucide Icons

## 📂 Project Structure

```text
├── main.py                # FastAPI Backend & Multi-Model API
├── submission_package/    # Organized files for academic submission
├── code/                  # Source scripts for all 5 algorithms
├── dataset/               # Raw and cleaned automotive records
├── presentation/          # Comparison reports and high-res plots
└── references/            # Project documentation and guides
```

## 🏁 Getting Started

### 1. Backend (Local API)
```bash
# Run the FastAPI server
./venv/bin/python3 main.py
```

### 2. Frontend (Dev Mode)
```bash
cd frontend
npm run dev
```

---
© 2026 Resale Intelligence Systems. Powered by Ensemble Modeling.
