# Global Car Resale Intelligence System

An advanced Machine Learning platform designed to predict vehicle resale values across global markets (India, Europe, and Asia/UK) using high-performance gradient boosting.

## 🚀 Key Features

- **XGBoost Neural Engine**: Utilizes L2 Gradient Boosting to analyze complex depreciation patterns.
- **Model-Specific Precision**: Differentiates valuation between specific vehicle models (e.g., Maruti Baleno vs. Maruti Swift) for maximum accuracy.
- **Multi-Market Capability**: Integrated data pipelines for India, Europe, and the UK.
- **Dynamic Multi-Currency**: Real-time conversion support for USD ($), INR (₹), EUR (€), and GBP (£).
- **High-Fidelity Dashboard**: A production-grade Glassmorphism interface built with React, Framer Motion, and Tailwind CSS.

## 📊 Performance Summary

The system addresses two core ML tasks with high reliability:

| Task | Metric | Performance |
|------|--------|-------------|
| **Regression** | R² Score | **88.0%** |
| **Classification** | Accuracy | **81.4%** |
| **Dataset Size** | Total Records | **364k+** |

Detailed metrics can be found in the [Model Evaluation Report](./model_evaluation_report.md).

## 🛠️ Tech Stack

- **Backend**: Python 3.9, FastAPI, Uvicorn
- **Machine Learning**: XGBoost, Scikit-learn, Pandas, Joblib
- **Frontend**: React 19, TypeScript, Vite, Framer Motion, Recharts, Lucide Icons
- **DevOps**: GitHub Pages (Frontend Hosting), Git

## 📂 Project Structure

```text
├── main.py                # FastAPI Backend & Prediction API
├── train_pipeline.py      # Data cleaning & ML training pipeline
├── xgb_pipeline.pkl       # Serialized XGBoost Regression Model
├── models_registry.json   # Hierarchical market/brand/model mapping
├── model_evaluation_report.md # Comprehensive ML audit report
├── frontend/              # React TypeScript Application
│   ├── src/App.tsx        # UI Logic & Visual Components
│   └── src/App.css        # Premium styling & Glassmorphism
└── plots/                 # Visual evidence (Confusion Matrix, Feature Importance)
```

## 🏁 Getting Started

### 1. Backend (Local API)
```bash
# Activate virtual environment
source venv/bin/activate
# Run the FastAPI server
python main.py
```

### 2. Frontend (Dev Mode)
```bash
cd frontend
npm install
npm run dev
```

## 📝 Note for Recruiters
This repository showcases a full-stack Machine Learning production lifecycle—from raw data cleaning and hyper-parameter tuning to a responsive, deployment-ready user interface. The frontend is hosted on **GitHub Pages** in Preview Mode, while the core API is designed for localized high-performance inference.

---
© 2026 Resale Intelligence Systems. Powered by XGBoost Gradient Boosting.
