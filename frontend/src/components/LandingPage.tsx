import { ArrowRight, Activity, Zap, Shield, ChevronRight } from 'lucide-react';

interface LandingPageProps {
  onEnterApp: () => void;
}

export function LandingPage({ onEnterApp }: LandingPageProps) {
  return (
    <div className="landing-premium-shell">
      {/* Background Ambient Glows */}
      <div className="ambient-glow glow-1"></div>
      <div className="ambient-glow glow-2"></div>

      {/* Hero Section */}
      <section className="hero-premium">
        <div className="hero-content-wrapper animate-fade-in-up">
          <div className="badge-premium">
            <span className="badge-dot"></span>
            Clinical Decision Support AI
          </div>
          <h1 className="hero-title-premium">
            Multimodal Lung Disease <br />
            <span className="text-gradient">Detection System</span>
          </h1>
          <p className="hero-subtitle-premium">
            Advanced AI-driven analysis combining chest imaging, respiratory audio, and clinical data 
            to support respiratory disease assessment with unprecedented precision.
          </p>
          <div className="hero-actions">
            <button className="btn-glow" onClick={onEnterApp}>
              Enter Clinical Interface
              <ArrowRight size={18} strokeWidth={2.5} />
            </button>
          </div>
        </div>
      </section>

      {/* Bento Grid Features Section */}
      <section className="features-bento animate-fade-in-up-delayed">
        <div className="bento-container">
          <div className="bento-header">
            <h2>Integrated Diagnostic Approach</h2>
            <p>Tri-modal intelligence for comprehensive respiratory analysis.</p>
          </div>
          
          <div className="bento-grid">
            <div className="bento-card span-2 image-branch">
              <div className="bento-icon-wrapper primary-bg">
                <Activity size={24} color="#ffffff" />
              </div>
              <h3>Imaging Analysis</h3>
              <p>Chest X-ray processing with deep learning models trained on large respiratory datasets. Acts as the primary evidence for disease classification.</p>
            </div>

            <div className="bento-card audio-branch">
              <div className="bento-icon-wrapper secondary-bg">
                <Zap size={24} color="#ffffff" />
              </div>
              <h3>Audio Auscultation</h3>
              <p>Mel-spectrogram extraction for pattern recognition.</p>
            </div>

            <div className="bento-card clinical-branch">
              <div className="bento-icon-wrapper support-bg">
                <Shield size={24} color="#ffffff" />
              </div>
              <h3>Clinical Context</h3>
              <p>Integration of vital signs and demographics.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Modern Workflow Section */}
      <section className="workflow-premium">
        <div className="workflow-container">
          <h2>How It Works</h2>
          <div className="workflow-track">
            <div className="workflow-node">
              <div className="node-circle">1</div>
              <h4>Upload Data</h4>
              <p>Provide X-ray, audio, and patient vitals.</p>
            </div>
            <div className="node-line"></div>
            <div className="workflow-node">
              <div className="node-circle">2</div>
              <h4>Process</h4>
              <p>Independent neural network analysis.</p>
            </div>
            <div className="node-line"></div>
            <div className="workflow-node">
              <div className="node-circle">3</div>
              <h4>Integrate</h4>
              <p>Structured clinical reporting generated.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Minimalist CTA Footer */}
      <footer className="footer-premium">
        <div className="footer-content">
          <h2>Ready to analyze cases?</h2>
          <p>Access the secure clinical interface now.</p>
          <button className="btn-glow-dark" onClick={onEnterApp}>
            Launch Application <ChevronRight size={18} />
          </button>
        </div>
        <div className="footer-disclaimer">
          <p>
            <strong>Clinical Disclaimer:</strong> This system is a decision support tool designed to augment clinical judgment, not replace it. 
            Final clinical decisions must incorporate complete patient history, physical examination, and specialist expertise.
          </p>
        </div>
      </footer>
    </div>
  );
}