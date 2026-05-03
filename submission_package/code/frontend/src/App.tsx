import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { AnimatePresence, motion } from 'framer-motion';
import {
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area
} from 'recharts';
import {
  Activity, ShieldCheck, Trophy
} from 'lucide-react';
import './App.css';

// Components
import NeuralHeatmap from './components/NeuralHeatmap';
import ErrorStability from './components/ErrorStability';
import FeatureHierarchy from './components/FeatureHierarchy';
import Leaderboard from './components/Leaderboard';
import ModelInspector from './components/ModelInspector';
import SectionTag from './components/UI/SectionTag';
import RevealSection from './components/UI/RevealSection';
import AnimatedCounter from './components/UI/AnimatedCounter';
import AnimatedTitle from './components/UI/AnimatedTitle';
import Logo from './components/UI/Logo';

/* ─── Fallback Registry ─── */
const FALLBACK_REGISTRY: any = {
  india: {
    maruti: ["800", "alto", "baleno", "brezza", "swift", "wagon"],
    tata: ["altroz", "harrier", "nexon", "punch", "tiago"],
    hyundai: ["creta", "i10", "i20", "verna", "venue"]
  },
  europe: {
    audi: ["a3", "a4", "a6", "q3", "q5"],
    bmw: ["3 series", "5 series", "x3", "x5"],
    vw: ["golf", "polo", "tiguan", "passat"]
  }
};

/* ═══════════════════════════════════════ */
/*              MAIN APP                  */
/* ═══════════════════════════════════════ */
function App() {
  const [market, setMarket] = useState<string>('india');
  const [registry, setRegistry] = useState<any>(FALLBACK_REGISTRY);
  const [formData, setFormData] = useState({
    market: 'india', brand: 'maruti', model: 'alto',
    year: 2018, mileage: 50000, fuel_type: 'petrol',
    transmission: 'manual', target_currency: 'AUTO'
  });
  const [result, setResult] = useState<{ price: number; segment: string; symbol: string } | null>(null);
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [, setConfigLoading] = useState(true);

  useEffect(() => { 
    fetchConfig(); 
    fetchMetrics(); 
    const interval = setInterval(fetchMetrics, 3000);
    return () => clearInterval(interval);
  }, []);

  const fetchConfig = async () => {
    try {
      setConfigLoading(true);
      const res = await axios.get('http://localhost:8000/config');
      if (res.data && Object.keys(res.data).length > 0) {
        setRegistry(res.data);
        const dm = Object.keys(res.data)[0];
        const db = Object.keys(res.data[dm])[0];
        const dmod = res.data[dm][db][0];
        setMarket(dm);
        setFormData(prev => ({ ...prev, market: dm, brand: db, model: dmod }));
      }
    } catch { console.log("Operating in Preview Mode."); }
    finally { setConfigLoading(false); }
  };

  const fetchMetrics = async () => {
    try { const res = await axios.get('http://localhost:8000/metrics'); setMetrics(res.data); }
    catch { }
  };

  const handleMarketChange = (m: string) => {
    setMarket(m);
    const b = Object.keys(registry[m])[0];
    setFormData({ ...formData, market: m, brand: b, model: registry[m][b][0] });
    setResult(null);
  };

  const handleBrandChange = (b: string) => {
    setFormData({ ...formData, brand: b, model: (registry[market][b] || [])[0] || '' });
  };

  const handlePredict = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setLoading(true);
    try {
      const r = await axios.post('http://localhost:8000/predict', formData);
      setResult({ price: r.data.predicted_price, segment: r.data.segment, symbol: r.data.symbol });
    } catch { console.error('Prediction engine offline.'); }
    finally { setLoading(false); }
  };

  const scrollTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
  };

  const chartData = [
    { name: 'R² Score', value: metrics?.regression?.r2 * 100 || 86.82 },
    { name: 'Accuracy', value: metrics?.classification?.accuracy * 100 || 81.7 },
    { name: 'Precision', value: metrics?.classification?.precision * 100 || 81.8 },
    { name: 'Recall', value: metrics?.classification?.recall * 100 || 81.7 },
    { name: 'F1 Score', value: metrics?.classification?.f1 * 100 || 81.7 },
  ];

  return (
    <>
      {/* ─── SITE FRAME ─── */}
      <div className="site-frame">
        <main>
          {/* ═══ HERO ═══ */}
          <section className="hero-section" id="hero">
            <div className="hero-bg">
              <div className="hero-bg-img" role="img" aria-label="Abstract neural network background" />
              <div className="hero-bg-gradient" />
            </div>

            <div className="hero-main">
              <motion.span
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.2 }}
                className="hero-tag"
              >
                system.26 // multi-algo engine active
              </motion.span>

              <h1 className="hero-title">
                <AnimatedTitle text="RESALE" delay={0.4} />
                <br />
                <AnimatedTitle text="INTELLIGENCE" className="blue-highlight" delay={0.8} />
              </h1>

              <motion.p
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 1.4 }}
                className="hero-desc"
              >
                Automotive valuation through ensemble modeling. Integrating XGBoost, Random Forest, and SVM for hyper-accurate price forecasting.
              </motion.p>
            </div>

            <div className="status-ticker">
              <div className="ticker-inner">
                <div className="ticker-item">
                  <div className="ticker-value">
                    <AnimatedCounter value={metrics?.regression?.r2 * 100 || 86.82} suffix="%" />
                  </div>
                  <div className="ticker-label">Max R² Score</div>
                </div>
                <div className="ticker-divider" />
                <div className="ticker-item">
                  <div className="ticker-value">{metrics?.dataset?.records ? `${Math.round(metrics.dataset.records / 1000)}K+` : '355K+'}</div>
                  <div className="ticker-label">Records Trained</div>
                </div>
                <div className="ticker-divider" />
                <div className="ticker-item">
                  <div className="ticker-value blue">ENSEMBLE</div>
                  <div className="ticker-label">5 Algorithms Online</div>
                </div>
              </div>
            </div>
          </section>

          {/* ═══ FORM SECTION ═══ */}
          <RevealSection className="form-section" id="variables">
            <SectionTag label="variables" />
            <div className="form-section-inner">
              <div className="form-header">
                <span className="form-indicator" />
                <h3>Input Parameters</h3>
              </div>

              <form onSubmit={handlePredict}>
                <div className="form-grid">
                  <div className="input-field-group span-2">
                    <label className="field-label">Market Region</label>
                    <nav className="market-nav" aria-label="Market selection">
                      {registry && Object.keys(registry).map((m) => (
                        <button
                          key={m} type="button"
                          className={`market-btn ${market === m ? 'active' : ''}`}
                          onClick={() => handleMarketChange(m)}
                          aria-pressed={market === m}
                        >
                          {m.replace('_', ' ')}
                        </button>
                      ))}
                    </nav>
                  </div>

                  <div className="input-field-group">
                    <label className="field-label" htmlFor="brand-select">Manufacturer</label>
                    <select id="brand-select" className="brand-select" value={formData.brand}
                      onChange={e => handleBrandChange(e.target.value)}>
                      {registry?.[market] && Object.keys(registry[market]).sort().map(b => (
                        <option key={b} value={b}>{b.toUpperCase()}</option>
                      ))}
                    </select>
                  </div>

                  <div className="input-field-group">
                    <label className="field-label" htmlFor="model-select">Model</label>
                    <select id="model-select" className="brand-select" value={formData.model}
                      onChange={e => setFormData({ ...formData, model: e.target.value })}>
                      {registry?.[market]?.[formData.brand]?.sort().map((m: string) => (
                        <option key={m} value={m}>{m.toUpperCase()}</option>
                      ))}
                    </select>
                  </div>

                  <div className="input-field-group">
                    <label className="field-label" htmlFor="year-input">Production Year</label>
                    <input id="year-input" type="number" className="brand-input" value={formData.year}
                      onChange={e => setFormData({ ...formData, year: Number(e.target.value) })} />
                  </div>

                  <div className="input-field-group">
                    <label className="field-label" htmlFor="mileage-input">Odometer (KM)</label>
                    <input id="mileage-input" type="number" className="brand-input" value={formData.mileage}
                      onChange={e => setFormData({ ...formData, mileage: Number(e.target.value) })} />
                  </div>
                </div>

                <button className="brand-btn" type="submit" disabled={loading} aria-busy={loading}>
                  {loading ? 'Analyzing Data...' : 'Execute Valuation  +'}
                </button>
              </form>
            </div>
          </RevealSection>

          {/* ═══ RESULT ═══ */}
          <AnimatePresence mode="wait">
            {result && (
              <motion.section
                className="result-section"
                id="analysis"
                initial={{ opacity: 0, y: 40 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
                aria-live="polite"
              >
                <SectionTag label="analysis" />
                <div className="result-card">
                  <div>
                    <span className="result-label">Market Equilibrium Value</span>
                    <div className="result-price">
                      <span className="result-symbol" aria-hidden="true">{result.symbol}</span>
                      {result.price.toLocaleString()}
                    </div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <span className="result-label">Classification</span>
                    <div className="result-segment">{result.segment.toUpperCase()}</div>
                  </div>
                </div>
              </motion.section>
            )}
          </AnimatePresence>

          {/* ═══ METRICS ═══ */}
          <RevealSection className="metrics-section" id="metrics">
            <SectionTag label="metrics" />
            <div className="metric-grid">
              <div className="metric-card">
                <div className="metric-value">
                  <AnimatedCounter value={metrics?.regression?.r2 * 100 || 86.82} suffix="%" />
                </div>
                <div className="metric-label">Ensemble R² Score</div>
              </div>
              <div className="metric-card">
                <div className="metric-value">
                  $<AnimatedCounter value={metrics?.regression?.mae || 3323} suffix="" decimals={0} />
                </div>
                <div className="metric-label">Mean Abs. Error</div>
              </div>
              <div className="metric-card">
                <div className="metric-value">
                  <AnimatedCounter value={metrics?.classification?.accuracy * 100 || 81.7} suffix="%" />
                </div>
                <div className="metric-label">Max Accuracy</div>
              </div>
              <div className="metric-card">
                <div className="metric-value">
                  $<AnimatedCounter value={metrics?.regression?.rmse || 6414} suffix="" decimals={0} />
                </div>
                <div className="metric-label">Ensemble RMSE</div>
              </div>
            </div>
          </RevealSection>

          {/* ═══ LEADERBOARD ═══ */}
          <RevealSection className="leaderboard-section" id="leaderboard">
            <SectionTag label="ranking" />
            <Leaderboard data={metrics?.leaderboard} />
          </RevealSection>

          {/* ═══ CHART ═══ */}
          <RevealSection className="chart-section" id="validation">
            <SectionTag label="validation" />
            <div className="chart-card">
              <div className="form-header">
                <Activity size={16} style={{ color: 'var(--accent-blue)' }} aria-hidden="true" />
                <h3>Validation Matrix</h3>
              </div>
              <div style={{ height: '420px' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
                    <XAxis dataKey="name" stroke="#444" fontSize={9} tickLine={false} axisLine={false}
                      fontFamily="'Press Start 2P'" />
                    <YAxis stroke="#444" fontSize={9} tickLine={false} axisLine={false} />
                    <Tooltip
                      contentStyle={{
                        background: 'rgba(0,0,0,0.9)', border: '1px solid rgba(255,255,255,0.1)',
                        color: '#fff', borderRadius: '8px', fontFamily: "'Inter'", fontSize: '0.8rem'
                      }}
                    />
                    <Area type="monotone" dataKey="value" stroke="var(--accent-blue)" strokeWidth={2}
                      fill="url(#blueGradient)" />
                    <defs>
                      <linearGradient id="blueGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="var(--accent-blue)" stopOpacity={0.15} />
                        <stop offset="100%" stopColor="var(--accent-blue)" stopOpacity={0.01} />
                      </linearGradient>
                    </defs>
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          </RevealSection>

          {/* ═══ AUDIT ═══ */}
          <RevealSection className="audit-section" id="audit">
            <SectionTag label="audit" />
            <div className="form-header">
              <ShieldCheck size={16} style={{ color: 'var(--accent-blue)' }} aria-hidden="true" />
              <h3>Ensemble Audit Trail</h3>
            </div>

            <div className="audit-grid">
              <div className="span-2" style={{ gridColumn: 'span 2' }}>
                 <ModelInspector leaderboardData={metrics?.leaderboard || []} />
              </div>

              <div className="audit-card-featured">
                <div className="plot-wrapper" style={{ padding: '3rem' }}>
                  <NeuralHeatmap data={metrics?.chart_data?.confusion} />
                </div>
                <div className="audit-overlay">
                  <div>
                    <div className="audit-card-label">Matrix 01</div>
                    <div className="audit-card-sublabel">XGBoost Clustering</div>
                  </div>
                </div>
              </div>

              <div className="audit-card-featured" style={{ background: 'rgba(255,255,255,0.01)' }}>
                <FeatureHierarchy data={metrics?.chart_data?.importance} />
                <div className="audit-overlay">
                  <div>
                    <div className="audit-card-label">Nodes 02</div>
                    <div className="audit-card-sublabel">Feature Hierarchy</div>
                  </div>
                </div>
              </div>

              <div className="audit-row" style={{ gridColumn: 'span 2' }}>
                <ErrorStability data={metrics?.chart_data?.residuals} metrics={metrics} />
              </div>
            </div>
          </RevealSection>
        </main>

        {/* ═══ FOOTER ═══ */}
        <RevealSection id="footer">
          <footer className="site-footer">
            <SectionTag label="sovereignty" />

            <h2 className="footer-title">
              DATA<br /><span className="blue-highlight">SOVEREIGNTY</span>
            </h2>

            <p className="footer-body">
              This interface operates as a high-fidelity demonstration of the global car resale intelligence platform.
              System utilizes a multi-algorithm ensemble (XGBoost, Random Forest, SVM, Decision Tree) trained on 364K+ multi-market records.
              Valuation logic is dynamically routed to the highest-performing model based on market segment volatility.
            </p>

            <div className="footer-divider" />

            <nav className="footer-nav" aria-label="Footer navigation">
              <button className="footer-nav-item" onClick={() => scrollTo('hero')}>Dashboard</button>
              <button className="footer-nav-item" onClick={() => scrollTo('variables')}>Variables</button>
              <button className="footer-nav-item" onClick={() => scrollTo('metrics')}>Metrics</button>
              <button className="footer-nav-item" onClick={() => scrollTo('leaderboard')}>Leaderboard</button>
              <button className="footer-nav-item" onClick={() => scrollTo('audit')}>Audit</button>
            </nav>

            <div className="footer-brand" aria-hidden="true">
              RESALE<br />INTELLIGENCE
            </div>

            <p className="footer-copy">© 2026 // Resale Intelligence // Professional Ensemble V6.0</p>
          </footer>
        </RevealSection>

      </div>

      {/* ─── FLOATING BOTTOM NAV ─── */}
      <nav className="floating-nav" aria-label="Main navigation">
        <div className="nav-logo">
          <Logo size={40} />
        </div>
        <div className="nav-links">
          <button className="nav-link" onClick={() => scrollTo('hero')}>Dashboard</button>
          <button className="nav-link" onClick={() => scrollTo('leaderboard')}>Ranking</button>
          <button className="nav-link" onClick={() => scrollTo('metrics')}>Metrics</button>
          <button className="nav-link" onClick={() => scrollTo('audit')}>Audit</button>
        </div>
        <button className="nav-cta" onClick={() => scrollTo('variables')}>Predict +</button>
      </nav>
    </>
  );
}

export default App;
