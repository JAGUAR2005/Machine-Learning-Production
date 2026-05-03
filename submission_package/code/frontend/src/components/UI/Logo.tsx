import React from 'react';

interface LogoProps {
  className?: string;
  size?: number | string;
}

const Logo: React.FC<LogoProps> = ({ className = '', size = '100%' }) => {
  return (
    <svg 
      className={className} 
      width={size} 
      height={size} 
      viewBox="0 0 200 200" 
      fill="none" 
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <linearGradient id="logoRingGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#40c9ff" />
          <stop offset="100%" stopColor="#0288d1" />
        </linearGradient>
        <linearGradient id="logoBarGrad" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="#40c9ff" />
          <stop offset="100%" stopColor="#0288d1" />
        </linearGradient>
        <radialGradient id="logoNodeGrad" cx="30%" cy="30%" r="70%">
          <stop offset="0%" stopColor="#ffffff" />
          <stop offset="40%" stopColor="#e2e8f0" />
          <stop offset="100%" stopColor="#94a3b8" />
        </radialGradient>
        <filter id="logoGlow" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
        <filter id="logoShadow" x="-10%" y="-10%" width="120%" height="120%">
          <feDropShadow dx="0" dy="2" stdDeviation="2" floodColor="#000000" floodOpacity="0.4" />
        </filter>
      </defs>

      {/* Outer Swooshes / Ring */}
      <g filter="url(#logoGlow)">
        {/* Left arc */}
        <path d="M 60 185 A 90 90 0 0 1 30 50" fill="none" stroke="url(#logoRingGrad)" strokeWidth="12" strokeLinecap="round" />
        
        {/* Top-Right arc */}
        <path d="M 70 20 A 90 90 0 0 1 180 80" fill="none" stroke="url(#logoRingGrad)" strokeWidth="12" strokeLinecap="round" />
        
        {/* Arrow head on the top-right arc */}
        <path d="M 160 60 L 180 80 L 165 100" fill="none" stroke="url(#logoRingGrad)" strokeWidth="12" strokeLinecap="round" strokeLinejoin="round" />
        
        {/* Bottom arc */}
        <path d="M 175 140 A 90 90 0 0 1 90 190" fill="none" stroke="url(#logoRingGrad)" strokeWidth="12" strokeLinecap="round" />
      </g>

      {/* Inner Graphics */}
      <g filter="url(#logoShadow)">
        {/* Vertical Bars */}
        <rect x="75" y="110" width="14" height="60" rx="7" fill="url(#logoBarGrad)" />
        <rect x="105" y="85" width="14" height="85" rx="7" fill="url(#logoBarGrad)" />
        <rect x="135" y="100" width="14" height="70" rx="7" fill="url(#logoBarGrad)" />

        {/* Small floating dots above bars */}
        <circle cx="82" cy="100" r="4.5" fill="url(#logoNodeGrad)" />
        <circle cx="112" cy="75" r="4.5" fill="url(#logoNodeGrad)" />
        <circle cx="142" cy="90" r="4.5" fill="url(#logoNodeGrad)" />

        {/* Network connections (lines) */}
        <line x1="112" y1="45" x2="65" y2="85" stroke="url(#logoRingGrad)" strokeWidth="4" />
        <line x1="112" y1="45" x2="160" y2="80" stroke="url(#logoRingGrad)" strokeWidth="4" />

        {/* Network Nodes (silver circles) */}
        <circle cx="112" cy="45" r="13" fill="url(#logoNodeGrad)" />
        <circle cx="65" cy="85" r="11" fill="url(#logoNodeGrad)" />
        <circle cx="160" cy="80" r="11" fill="url(#logoNodeGrad)" />
      </g>
    </svg>
  );
};

export default Logo;
