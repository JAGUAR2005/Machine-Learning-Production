import React from 'react';
import { Layers, Globe, Zap, Network } from 'lucide-react';

const ArchitectureShowcase: React.FC = () => {
  const plots = [
    {
      id: 'svm',
      title: 'SVM Hyperplane',
      subtitle: '3D Depreciation Surface',
      url: 'http://localhost:8000/plots/svm/svm_hyperplane_3d.png',
      icon: <Globe size={14} />,
      desc: 'Visualizing the Support Vector Machine high-dimensional boundary for value decay.'
    },
    {
      id: 'dt',
      title: 'Logic Flow',
      subtitle: 'Decision Tree Logic Map',
      url: 'http://localhost:8000/plots/decision_tree/actual_tree_map.png',
      icon: <Network size={14} />,
      desc: 'Actual branching logic used to categorize vehicles based on age and mileage.'
    },
    {
      id: 'rf',
      title: 'Forest Impact',
      subtitle: 'Collective Wisdom Map',
      url: 'http://localhost:8000/plots/random_forest/forest_feature_impact.png',
      icon: <Zap size={14} />,
      desc: 'Aggregate feature influence across 100 independent decision trees.'
    }
  ];

  return (
    <div className="architecture-showcase">
      <div className="form-header">
        <Layers size={16} style={{ color: 'var(--accent-blue)' }} />
        <h3>Model Architecture Showcase</h3>
      </div>
      
      <div className="showcase-grid">
        {plots.map(plot => (
          <div key={plot.id} className="showcase-card">
            <div className="showcase-img-container">
              <img src={plot.url} alt={plot.title} className="showcase-img" />
              <div className="showcase-badge">
                {plot.icon}
                <span>{plot.id.toUpperCase()}</span>
              </div>
            </div>
            <div className="showcase-content">
              <div className="showcase-title-group">
                <span className="showcase-subtitle">{plot.subtitle}</span>
                <h4 className="showcase-title">{plot.title}</h4>
              </div>
              <p className="showcase-desc">{plot.desc}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ArchitectureShowcase;
