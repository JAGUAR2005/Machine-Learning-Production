import { useState, useEffect, useRef } from 'react';
import { motion, useInView } from 'framer-motion';

interface ErrorStabilityProps {
  data: any[];
  metrics: any;
}

const ErrorStability = ({ data, metrics }: ErrorStabilityProps) => {
  const [lockedData, setLockedData] = useState<any[] | null>(null);
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-40px" });

  useEffect(() => {
    if (isInView && data && !lockedData) {
      setLockedData(data);
    }
  }, [isInView, data, lockedData]);

  if (!data) return <div className="loading-placeholder">Calibrating Residuals...</div>;

  const displayData = lockedData || data;

  const width = 720;
  const height = 280;
  const paddingX = 44;
  const centerY = 133; 
  const maxPrice = 50000;
  const maxError = 3000;
  
  const mae = metrics?.regression?.mae || 3463;
  const maePx = (mae / maxError) * 100;

  return (
    <div ref={ref} className="glass-strong rounded-xl p-6 relative scanline overflow-hidden" role="img" aria-label="Error Stability Plot showing model residuals across price ranges">
      <div className="flex items-center justify-between mb-4">
        <div className="font-pixel text-[10px] tracking-widest text-cobalt">◆ ERROR_STABILITY</div>
        <div className="font-mono-num text-[10px] text-muted-foreground">
          MAE ±{(mae / 1000).toFixed(2)}k · n={displayData.length}
        </div>
      </div>
      
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto">
        <defs>
          <radialGradient id="dotGrad">
            <stop offset="0%" stopColor="#3b82f6" stopOpacity={1} />
            <stop offset="60%" stopColor="#1e3a8a" stopOpacity={0.8} />
            <stop offset="100%" stopColor="#1e3a8a" stopOpacity={0} />
          </radialGradient>
        </defs>

        {/* Horizontal Grid */}
        <line x1={paddingX} x2={704} y1={centerY + 100} y2={centerY + 100} stroke="rgba(255,255,255,0.05)" strokeDasharray="2 4" />
        <line x1={paddingX} x2={704} y1={centerY + 66} y2={centerY + 66} stroke="rgba(255,255,255,0.05)" strokeDasharray="2 4" />
        <line x1={paddingX} x2={704} y1={centerY + 33} y2={centerY + 33} stroke="rgba(255,255,255,0.05)" strokeDasharray="2 4" />
        <line x1={paddingX} x2={704} y1={centerY} y2={centerY} stroke="rgba(59, 130, 246, 0.4)" />
        <line x1={paddingX} x2={704} y1={centerY - 33} y2={centerY - 33} stroke="rgba(255,255,255,0.05)" strokeDasharray="2 4" />
        <line x1={paddingX} x2={704} y1={centerY - 66} y2={centerY - 66} stroke="rgba(255,255,255,0.05)" strokeDasharray="2 4" />
        <line x1={paddingX} x2={704} y1={centerY - 100} y2={centerY - 100} stroke="rgba(255,255,255,0.05)" strokeDasharray="2 4" />

        {/* Vertical Grid */}
        {[0, 10, 20, 30, 40, 50].map((v, i) => (
          <line key={v} x1={paddingX + i * 132} x2={paddingX + i * 132} y1={16} y2={250} stroke="rgba(255,255,255,0.04)" />
        ))}

        {/* Confidence Zone (MAE) */}
        <rect x={paddingX} y={centerY - maePx} width={660} height={maePx * 2} fill="rgba(59, 130, 246, 0.07)" />
        <line x1={paddingX} x2={704} y1={centerY - maePx} y2={centerY - maePx} stroke="rgba(59, 130, 246, 0.3)" strokeDasharray="4 4" />
        <line x1={paddingX} x2={704} y1={centerY + maePx} y2={centerY + maePx} stroke="rgba(59, 130, 246, 0.3)" strokeDasharray="4 4" />

        {/* Axis Labels */}
        <text x="36" y={centerY + 103} textAnchor="end" fontSize="9" fill="#666" fontFamily="'JetBrains Mono', monospace">-3k</text>
        <text x="36" y={centerY + 3} textAnchor="end" fontSize="9" fill="#666" fontFamily="'JetBrains Mono', monospace">0k</text>
        <text x="36" y={centerY - 97} textAnchor="end" fontSize="9" fill="#666" fontFamily="'JetBrains Mono', monospace">+3k</text>
        
        {[0, 10, 20, 30, 40, 50].map((v, i) => (
          <text key={v} x={paddingX + i * 132} y="268" textAnchor="middle" fontSize="9" fill="#666" fontFamily="'JetBrains Mono', monospace">{v}k</text>
        ))}

        {/* Dynamic Data Points */}
        {displayData.map((point, i) => {
          const cx = paddingX + (point.actual / maxPrice) * 660;
          const cy = centerY - (point.residual / maxError) * 100;
          return (
            <motion.circle
              key={i}
              cx={cx}
              cy={cy}
              r={3.5}
              fill="url(#dotGrad)"
              initial={{ opacity: 0, scale: 0 }}
              animate={{ 
                opacity: [0.2, 0.9, 0.2],
                scale: [0.8, 1.2, 0.8]
              }}
              transition={{ 
                duration: 2 + Math.random() * 2,
                repeat: Infinity,
                delay: Math.random() * 2
              }}
            />
          );
        })}
      </svg>
    </div>
  );
};

export default ErrorStability;
