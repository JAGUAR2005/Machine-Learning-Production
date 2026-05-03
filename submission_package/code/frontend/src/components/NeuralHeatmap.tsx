import React from 'react';
import { motion } from 'framer-motion';

interface NeuralHeatmapProps {
  data: any[];
}

const NeuralHeatmap = ({ data }: NeuralHeatmapProps) => {
  if (!data) return <div className="loading-placeholder">Initializing Matrix...</div>;
  
  const categories = ["Budget", "Mid", "Prem", "Lux"];
  
  return (
    <div className="heatmap-container">
      <div className="heatmap-grid">
        <div className="heatmap-corner"></div>
        {categories.map(c => <div key={c} className="heatmap-label-top">{c}</div>)}
        
        {data.map((row, i) => (
          <React.Fragment key={i}>
            <div className="heatmap-label-left">{categories[i]}</div>
            {categories.map(cat => {
              const val = row[cat];
              const intensity = val / 100;
              return (
                <motion.div 
                  key={cat}
                  className="heatmap-cell"
                  initial={{ opacity: 0, scale: 0.8 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  transition={{ delay: (i * categories.length + categories.indexOf(cat)) * 0.03 }}
                  style={{ 
                    background: `rgba(59, 130, 246, ${intensity * 0.8 + 0.05})`,
                    border: `1px solid rgba(59, 130, 246, ${intensity * 0.2})`
                  }}
                  aria-label={`Correlation between ${categories[i]} and ${cat}: ${val}%`}
                >
                  <span className="heatmap-val">{val}%</span>
                </motion.div>
              );
            })}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
};

export default NeuralHeatmap;
