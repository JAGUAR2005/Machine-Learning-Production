import React from 'react';
import { motion } from 'framer-motion';
import { Trophy, TrendingUp, BarChart } from 'lucide-react';

interface LeaderboardProps {
  data: any[];
}

const Leaderboard: React.FC<LeaderboardProps> = ({ data }) => {
  if (!data || data.length === 0) return null;

  return (
    <div className="leaderboard-container">
      <div className="form-header">
        <Trophy size={16} style={{ color: 'var(--accent-blue)' }} />
        <h3>Algorithm Leaderboard</h3>
      </div>
      
      <div className="leaderboard-table-wrapper">
        <table className="leaderboard-table">
          <thead>
            <tr>
              <th>RANK</th>
              <th>ALGORITHM</th>
              <th>PRICE ACCURACY (R²)</th>
              <th>SEGMENTATION</th>
              <th>STATUS</th>
            </tr>
          </thead>
          <tbody>
            {data.map((item, index) => (
              <motion.tr 
                key={item.algorithm}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className={item.is_primary ? 'row-highlight' : ''}
              >
                <td className="rank-col">#0{index + 1}</td>
                <td className="algo-col">
                    <span className="algo-name">{item.algorithm}</span>
                    {item.is_primary && <span className="primary-tag">PRODUCTION</span>}
                </td>
                <td className="metric-col">
                    <div className="metric-bar-bg">
                        <motion.div 
                            className="metric-bar-fill"
                            initial={{ width: 0 }}
                            animate={{ width: `${item.r2 * 100}%` }}
                            transition={{ duration: 1, delay: 0.5 }}
                        />
                    </div>
                    <span className="metric-value">{(item.r2 * 100).toFixed(2)}%</span>
                </td>
                <td className="metric-col">
                    <span className="metric-value">{(item.accuracy * 100).toFixed(1)}%</span>
                </td>
                <td>
                    <div className="status-indicator">
                        <div className="dot" />
                        VALIDATED
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

export default Leaderboard;
