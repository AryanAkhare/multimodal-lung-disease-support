export interface TabularInputs {
  age: number;
  gender: string;
  smoking_status: string;
  fev1_percent: number;
  spo2: number;
  respiratory_rate: number;
  cough_severity: number;
  wheeze: string;
  chest_tightness: string;
  crackles: string;
  fever: string;
  bmi: number;
  copd_exacerbations: number;
}

export interface BranchScore {
  label: string;
  confidence: number;
  message: string;
}

export interface FinalReport {
  predicted_class: string;
  overall_confidence: number;
  recommendation: string[];
  note: string;
}

export interface PredictionResponse {
  image_branch: BranchScore;
  audio_branch: BranchScore;
  tabular_branch: BranchScore;
  final_report: FinalReport;
}