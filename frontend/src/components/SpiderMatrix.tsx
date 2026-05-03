import React from 'react';
import {
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer
} from 'recharts';
import { motion } from 'framer-motion';
import { ShieldCheck, Target } from 'lucide-react';

interface SpiderMatrixProps {
  data: any[];
}

const SpiderMatrix: React.FC<SpiderMatrixProps> = ({ data }) => {
  if (!data || data.length === 0) return null;

  // Normalize data for the spider chart (0-100 scale)
  // For MAE and RMSE, lower is better, so we invert the value
  const maxMAE = 6000;
  const maxRMSE = 10000;

  const prepareData = (algo: any) => {
    return [
      { subject: 'R² Score', value: algo.r2 * 100, fullMark: 100 },
      { subject: 'Accuracy', value: algo.accuracy * 100, fullMark: 100 },
      { subject: 'Precision', value: Math.max(0, 100 - (algo.mae / maxMAE) * 100), fullMark: 100 },
      { subject: 'Stability', value: Math.max(0, 100 - (algo.rmse / maxRMSE) * 100), fullMark: 100 },
    ];
  };

  const getAlgoColor = (index: number) => {
    const colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444'];
    return colors[index % colors.length];
  };

  return (
    <div className="spider-matrix-container">
      <div className="form-header">
        <Target size={16} style={{ color: 'var(--accent-blue)' }} />
        <h3>Model Performance Spider Matrix</h3>
      </div>

      <div className="spider-grid">
        {data.map((algo, index) => (
          <motion.div 
            key={algo.algorithm}
            className="spider-card"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: index * 0.1 }}
          >
            <div className="spider-card-header">
              <span className="spider-rank">#0{index + 1}</span>
              <h4 className="spider-title">{algo.algorithm}</h4>
            </div>

            <div className="spider-chart-wrapper">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart cx="50%" cy="50%" outerRadius="85%" data={prepareData(algo)}>
                  <PolarGrid stroke="rgba(255,255,255,0.1)" />
                  <PolarAngleAxis 
                    dataKey="subject" 
                    tick={{ fill: 'rgba(255,255,255,0.8)', fontSize: 11, fontFamily: 'IBM Plex Mono', fontWeight: 600 }} 
                  />
                  <Radar
                    name={algo.algorithm}
                    dataKey="value"
                    stroke={getAlgoColor(index)}
                    fill={getAlgoColor(index)}
                    fillOpacity={0.6}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            <div className="spider-stats">
                <div className="spider-stat-item">
                    <span className="stat-label">R²</span>
                    <span className="stat-value">{(algo.r2).toFixed(3)}</span>
                </div>
                <div className="spider-stat-item">
                    <span className="stat-label">ACC</span>
                    <span className="stat-value">{(algo.accuracy * 100).toFixed(1)}%</span>
                </div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
};

export default SpiderMatrix;
