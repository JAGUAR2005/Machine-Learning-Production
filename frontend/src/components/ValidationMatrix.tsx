import React from 'react';
import { motion } from 'framer-motion';
import { Database, ShieldCheck, Zap } from 'lucide-react';

interface ValidationMatrixProps {
  data: any[];
}

const ValidationMatrix: React.FC<ValidationMatrixProps> = ({ data }) => {
  if (!data || data.length === 0) return null;

  return (
    <div className="validation-matrix-container">
      <div className="form-header">
        <ShieldCheck size={16} style={{ color: 'var(--accent-blue)' }} />
        <h3>Full Validation Matrix (Comparison)</h3>
      </div>
      
      <div className="matrix-grid-scroll">
        <table className="matrix-table">
          <thead>
            <tr>
              <th>ALGORITHM</th>
              <th>R² SCORE</th>
              <th>ACCURACY</th>
              <th>MAE (ERR)</th>
              <th>RMSE</th>
              <th>CONFIDENCE</th>
            </tr>
          </thead>
          <tbody>
            {data.map((algo, index) => (
              <motion.tr 
                key={algo.algorithm}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
              >
                <td className="matrix-algo-name">
                  <div className="flex items-center gap-2">
                    {algo.is_primary ? <Zap size={12} className="text-cobalt" /> : <Database size={12} />}
                    {algo.algorithm}
                  </div>
                </td>
                <td className="matrix-val mono">{(algo.r2).toFixed(4)}</td>
                <td className="matrix-val mono">{(algo.accuracy * 100).toFixed(2)}%</td>
                <td className="matrix-val mono">${Math.round(algo.mae).toLocaleString()}</td>
                <td className="matrix-val mono">${Math.round(algo.rmse).toLocaleString()}</td>
                <td>
                    <div className={`reliability-tag ${algo.r2 > 0.85 ? 'high' : algo.r2 > 0.75 ? 'med' : 'low'}`}>
                        {algo.r2 > 0.85 ? 'OPTIMAL' : algo.r2 > 0.75 ? 'STABLE' : 'BASELINE'}
                    </div>
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default ValidationMatrix;
