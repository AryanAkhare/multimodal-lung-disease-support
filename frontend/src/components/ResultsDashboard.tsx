import { ClipboardList, AlertCircle, CheckCircle2 } from 'lucide-react';
import type { PredictionResponse } from '../types';
import { BranchCard } from './BranchCard';

interface ResultsDashboardProps {
  loading: boolean;
  prediction: PredictionResponse | null;
}

export function ResultsDashboard({ loading, prediction }: ResultsDashboardProps) {
  // Premium Loading Skeleton State
  if (loading) {
    return (
      <div className="result-premium-grid animate-fade-in-up">
        <div className="branch-premium-grid">
          <div className="skeleton-card skeleton-pulse"></div>
          <div className="skeleton-card skeleton-pulse animation-delay-1"></div>
        </div>
        <div className="skeleton-card skeleton-pulse animation-delay-2" style={{ height: '120px' }}></div>
        <div className="skeleton-card skeleton-pulse animation-delay-3" style={{ height: '300px' }}></div>
      </div>
    );
  }

  // Premium Empty State
  if (!prediction) {
    return (
      <div className="empty-state-premium animate-fade-in-up">
        <div className="empty-state-icon">
          <ClipboardList size={40} strokeWidth={1.5} />
        </div>
        <h3>Awaiting Patient Data</h3>
        <p>Upload imaging and audio, then run the analysis to generate a comprehensive multimodal clinical report.</p>
      </div>
    );
  }

  // Loaded State
  return (
    <div className="result-premium-grid animate-fade-in-up">
      {/* Branch Analysis Grid */}
      <div className="branch-premium-grid">
        <BranchCard title="Imaging Evidence" score={prediction.image_branch} accent="primary" />
        <BranchCard title="Audio Auscultation" score={prediction.audio_branch} accent="secondary" />
      </div>

      <BranchCard title="Clinical & Tabular Context" score={prediction.tabular_branch} accent="support" />

      {/* Structured Medical Report */}
      <section className="report-premium-block">
        <div className="report-header">
          <div className="report-badge">
            <AlertCircle size={16} />
            AI Synthesis Report
          </div>
          <h3>Clinical Summary</h3>
        </div>

        <div className="report-detail-grid">
          <div className="report-row highlight-row">
            <span className="row-label">Primary Finding</span>
            <span className="row-value finding-text">{prediction.final_report.primary_finding}</span>
          </div>
          
          <div className="report-row">
            <span className="row-label">Diagnostic Confidence</span>
            <span className="row-value confidence-badge">
              {prediction.final_report.overall_confidence.toFixed(1)}%
            </span>
          </div>
          
          <div className="report-row block-row">
            <span className="row-label">Clinical Note</span>
            <p className="row-value text-block">{prediction.final_report.note}</p>
          </div>
        </div>

        <div className="recommendation-premium-section">
          <h4 className="rec-title">Recommended Clinical Actions</h4>
          <ul className="rec-list">
            {prediction.final_report.recommendation.map((item, index) => (
              <li key={index} className="rec-item">
                <CheckCircle2 size={18} className="rec-icon" />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>
      </section>
    </div>
  );
}