import { ChangeEvent } from 'react';
import { Mic, ImagePlus, CheckCircle2, FileImage, FileAudio } from 'lucide-react';

interface UploadPanelProps {
  imageFile: File | null;
  audioFile: File | null;
  onImageChange: (file: File | null) => void;
  onAudioChange: (file: File | null) => void;
}

export function UploadPanel({ imageFile, audioFile, onImageChange, onAudioChange }: UploadPanelProps) {
  return (
    <div className="upload-premium-panel">
      {/* Image Upload Zone */}
      <div className="form-premium-row">
        <label>Chest X-ray Imaging</label>
        <div className={`dropzone-premium ${imageFile ? 'state-filled' : 'state-empty'}`}>
          <input
            type="file"
            accept="image/*"
            className="dropzone-input"
            onChange={(event: ChangeEvent<HTMLInputElement>) => onImageChange(event.target.files?.[0] ?? null)}
          />
          <div className="dropzone-content">
            {imageFile ? (
              <>
                <div className="icon-success"><CheckCircle2 size={28} /></div>
                <strong className="file-name">{imageFile.name}</strong>
                <span className="file-size">{(imageFile.size / 1024 / 1024).toFixed(2)} MB • Ready for analysis</span>
              </>
            ) : (
              <>
                <div className="icon-idle primary-tint"><ImagePlus size={28} /></div>
                <strong>Upload lung image</strong>
                <span>Drag & drop or click to browse</span>
                <span className="hint">PNG/JPG (Recommended 224x224)</span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Audio Upload Zone */}
      <div className="form-premium-row">
        <label>Respiratory Auscultation</label>
        <div className={`dropzone-premium ${audioFile ? 'state-filled' : 'state-empty'}`}>
          <input
            type="file"
            accept="audio/*"
            className="dropzone-input"
            onChange={(event: ChangeEvent<HTMLInputElement>) => onAudioChange(event.target.files?.[0] ?? null)}
          />
          <div className="dropzone-content">
            {audioFile ? (
              <>
                <div className="icon-success"><CheckCircle2 size={28} /></div>
                <strong className="file-name">{audioFile.name}</strong>
                <span className="file-size">{(audioFile.size / 1024 / 1024).toFixed(2)} MB • Ready for analysis</span>
              </>
            ) : (
              <>
                <div className="icon-idle secondary-tint"><Mic size={28} /></div>
                <strong>Upload audio recording</strong>
                <span>Drag & drop or click to browse</span>
                <span className="hint">WAV or MP3 format</span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Clinical Legend */}
      <div className="clinical-legend">
        <h4>Diagnostic Evidence Legend</h4>
        <div className="legend-grid">
          <div className="legend-item">
            <span className="dot dot-primary"></span>
            <div>
              <strong>Image Branch</strong>
              <span>Primary spatial evidence</span>
            </div>
          </div>
          <div className="legend-item">
            <span className="dot dot-secondary"></span>
            <div>
              <strong>Audio Branch</strong>
              <span>Auxiliary acoustic patterns</span>
            </div>
          </div>
          <div className="legend-item">
            <span className="dot dot-support"></span>
            <div>
              <strong>Tabular Branch</strong>
              <span>Supportive patient context</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}