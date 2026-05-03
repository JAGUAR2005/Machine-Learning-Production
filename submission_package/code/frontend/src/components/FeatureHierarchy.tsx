import { useState, useEffect, useRef } from 'react';
import { motion, useInView } from 'framer-motion';
import { ResponsiveContainer, BarChart, XAxis, YAxis, Bar, Cell, LabelList } from 'recharts';

interface FeatureHierarchyProps {
  data: any[];
}

const FeatureHierarchy = ({ data }: FeatureHierarchyProps) => {
  const [lockedData, setLockedData] = useState<any[] | null>(null);
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-40px" });

  useEffect(() => {
    if (isInView && data && !lockedData) {
      setLockedData(data);
    }
  }, [isInView, data, lockedData]);

  if (!data) return <div className="loading-placeholder">Mapping Hierarchy...</div>;

  const displayData = lockedData || data;

  return (
    <div ref={ref} className="glass-strong rounded-xl p-6 relative scanline overflow-hidden" role="img" aria-label="Feature Hierarchy Chart showing importance of different variables">
      <div className="flex items-center justify-between mb-4">
        <div className="font-pixel text-[10px] tracking-widest text-cobalt">◆ FEATURE_HIERARCHY</div>
        <div className="font-mono-num text-[10px] text-muted-foreground">n={displayData.length} Nodes Active</div>
      </div>
      
      <div style={{ height: '220px', width: '100%' }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={displayData} layout="vertical" margin={{ left: -20, right: 20 }}>
            <XAxis type="number" hide domain={[0, 100]} />
            <YAxis dataKey="name" type="category" stroke="#666" fontSize={8} tickLine={false} axisLine={false} fontFamily="'JetBrains Mono', monospace" />
            <Bar dataKey="value" barSize={12}>
              {displayData.map((_entry, index) => {
                const gradientId = `barGrad-${index}`;
                return (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={`url(#${gradientId})`}
                  />
                );
              })}
              <LabelList dataKey="value" position="right" fill="#888" fontSize={8} fontFamily="'JetBrains Mono', monospace" formatter={(v: any) => v.toFixed(1)} />
            </Bar>
            <defs>
              {displayData.map((_, index) => (
                <linearGradient key={index} id={`barGrad-${index}`} x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.4} />
                  <motion.stop 
                    offset="50%" 
                    stopColor="#fff" 
                    animate={{ offset: ["0%", "100%", "0%"] }}
                    transition={{ duration: 3, repeat: Infinity, ease: "linear", delay: index * 0.2 }}
                    stopOpacity={0.6} 
                  />
                  <stop offset="100%" stopColor="#1e3a8a" stopOpacity={0.8} />
                </linearGradient>
              ))}
            </defs>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default FeatureHierarchy;
