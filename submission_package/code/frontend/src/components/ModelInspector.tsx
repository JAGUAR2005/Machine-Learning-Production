import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Layout, BarChart3, Binary, Network, Globe, Zap, Info } from 'lucide-react';

interface ModelInspectorProps {
  leaderboardData: any[];
}

const ModelInspector: React.FC<ModelInspectorProps> = ({ leaderboardData }) => {
  const [selectedAlgo, setSelectedAlgo] = useState(leaderboardData[0]?.algorithm || 'XGBoost (Optimized)');

  const algoDetails: any = {
    'XGBoost (Optimized)': {
      icon: <Zap size={20} />,
      graph: 'http://localhost:8000/plots/predicted_vs_actual.png',
      description: 'Gradient Boosting using decision trees with L2 regularization. Best for handling non-linear relationships and market segmentation.',
      features: ['L2 Regularization', 'Handling Missing Values', 'High Speed']
    },
    'Random Forest': {
      icon: <Binary size={20} />,
      graph: 'http://localhost:8000/plots/random_forest/single_tree_from_forest.png',
      description: 'An ensemble of 100 independent decision trees. Superior stability and accuracy in exact price forecasting.',
      features: ['Bagging Method', 'Reduced Overfitting', 'Feature Randomness']
    },
    'Decision Tree': {
      icon: <Network size={20} />,
      graph: 'http://localhost:8000/plots/decision_tree/actual_tree_map.png',
      description: 'A single, high-fidelity decision logic map. Highly interpretable but prone to slight variance.',
      features: ['Pure Logic Splitting', 'Zero Preprocessing', 'Visual Traceability']
    },
    'Support Vector Machine': {
      icon: <Globe size={20} />,
      graph: 'http://localhost:8000/plots/svm/svm_hyperplane_3d.png',
      description: 'Support Vector Regression (SVR) finding a hyperplane in high-dimensional space. Maps the complex depreciation surface.',
      features: ['Kernel Mapping', 'Margin Optimization', 'Geometric Stability']
    },
    'Logistic/Linear Regression': {
      icon: <BarChart3 size={20} />,
      graph: 'http://localhost:8000/plots/logistic/predicted_vs_actual.png',
      description: 'The baseline statistical model. Uses a weighted sum of input features to estimate price.',
      features: ['Linear Coefficients', 'Fast Execution', 'Statistical Baseline']
    }
  };

  const current = leaderboardData.find(a => a.algorithm === selectedAlgo) || leaderboardData[0];
  const details = algoDetails[current?.algorithm] || algoDetails['XGBoost (Optimized)'];

  return (
    <div className="model-inspector">
      <div className="form-header">
        <Layout size={16} style={{ color: 'var(--accent-blue)' }} />
        <h3>Algorithm Deep-Dive Inspector</h3>
      </div>

      <div className="inspector-layout">
        {/* Sidebar Nav */}
        <div className="inspector-sidebar">
          {leaderboardData.map((algo) => (
            <button
              key={algo.algorithm}
              className={`inspector-nav-item ${selectedAlgo === algo.algorithm ? 'active' : ''}`}
              onClick={() => setSelectedAlgo(algo.algorithm)}
            >
              <span className="nav-icon">{algoDetails[algo.algorithm]?.icon}</span>
              <div className="nav-text">
                <span className="nav-title">{algo.algorithm}</span>
                <span className="nav-stat">R²: {(algo.r2 * 100).toFixed(1)}%</span>
              </div>
            </button>
          ))}
        </div>

        {/* Display Area */}
        <div className="inspector-display">
          <AnimatePresence mode="wait">
            <motion.div
              key={selectedAlgo}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.3 }}
              className="inspector-content"
            >
              <div className="inspector-graph-view">
                <div className="graph-label">ACTUAL OUTPUT VISUALIZATION</div>
                <img src={details.graph} alt={selectedAlgo} className="inspector-img" />
                <div className="graph-zoom-tip">High-Resolution Training Result</div>
              </div>

              <div className="inspector-info">
                <div className="info-header">
                  <h4>{selectedAlgo}</h4>
                  <div className="performance-chip">
                    ACCURACY: {(current.accuracy * 100).toFixed(1)}%
                  </div>
                </div>
                
                <p className="info-desc">{details.description}</p>
                
                <div className="info-tags">
                  {details.features.map((f: string) => (
                    <span key={f} className="info-tag">◆ {f}</span>
                  ))}
                </div>

                <div className="info-metrics-row">
                    <div className="mini-metric">
                        <span className="mini-label">R² VALUE</span>
                        <span className="mini-value">{(current.r2).toFixed(4)}</span>
                    </div>
                    <div className="mini-metric">
                        <span className="mini-label">RELIABILITY</span>
                        <span className="mini-value">{current.r2 > 0.8 ? 'HIGH' : 'MEDIUM'}</span>
                    </div>
                </div>
              </div>
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
};

export default ModelInspector;
