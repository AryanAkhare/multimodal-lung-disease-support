import { Activity, Zap, Shield } from 'lucide-react';
import type { BranchScore } from '../types';

interface BranchCardProps {
  title: string;
  score: BranchScore;
  accent: 'primary' | 'secondary' | 'support';
}

export function BranchCard({ title, score, accent }: BranchCardProps) {
  // Map the accent type to a specific icon for better visual hierarchy
  const getIcon = () => {
    switch (accent) {
      case 'primary': return <Activity size={18} strokeWidth={2.5} />;
      case 'secondary': return <Zap size={18} strokeWidth={2.5} />;
      case 'support': return <Shield size={18} strokeWidth={2.5} />;
    }
  };

  return (
    <div className={`branch-premium-card ${accent}`}>
      {/* Background hover glow effect */}
      <div className="card-bg-glow"></div>
      
      <div className="card-content">
        <div className="card-header-premium">
          <div className={`icon-badge ${accent}`}>
            {getIcon()}
          </div>
          <h3>{title}</h3>
        </div>
        
        <div className="card-body">
          <p className="message">{score.message}</p>
        </div>
        
        <div className="card-footer">
          <div className="score-details">
            <span className="score-label">{score.label}</span>
            <span className="confidence-text">{score.confidence.toFixed(1)}% Conf</span>
          </div>
          
          {/* Animated Progress Bar */}
          <div className="progress-track">
            <div 
              className={`progress-fill ${accent}`} 
              style={{ width: `${Math.max(5, score.confidence)}%` }} /* Minimum 5% so the bar is always visible */
            >
              <div className="progress-glow"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}