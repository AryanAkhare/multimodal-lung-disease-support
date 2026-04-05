import type { PredictionResponse, TabularInputs } from '../types';

interface PredictRequest {
  imageFile: File | null;
  audioFile: File | null;
  tabular: TabularInputs;
}

export async function predictPatientData(request: PredictRequest): Promise<PredictionResponse> {
  const formData = new FormData();

  if (request.imageFile) {
    formData.append('image', request.imageFile);
  }
  if (request.audioFile) {
    formData.append('audio', request.audioFile);
  }

  Object.entries(request.tabular).forEach(([key, value]) => {
    formData.append(key, String(value));
  });

  const response = await fetch('http://localhost:8000/api/predict', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }

  const data: PredictionResponse = await response.json();
  return data;
}
