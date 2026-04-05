import type { PredictionResponse, TabularInputs } from '../types';

interface PredictRequest {
  imageFile: File | null;
  audioFile: File | null;
  tabular: TabularInputs;
}

const BASE_URL = import.meta.env.VITE_API_URL;

export async function checkApiHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${BASE_URL}/health`);
    return response.ok;
  } catch {
    return false;
  }
}

export async function predictPatientData(request: PredictRequest): Promise<PredictionResponse> {
  // Validate at least image or audio is provided
  if (!request.imageFile && !request.audioFile) {
    throw new Error('Please provide at least an image or audio file');
  }

  const formData = new FormData();

  if (request.imageFile) {
    formData.append('image', request.imageFile);
  }
  if (request.audioFile) {
    formData.append('audio', request.audioFile);
  }

  // Append all tabular fields
  Object.entries(request.tabular).forEach(([key, value]) => {
    formData.append(key, String(value));
  });

  try {
    const response = await fetch(`${BASE_URL}/api/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || `API error: ${response.statusText}`);
    }

    const data: PredictionResponse = await response.json();
    return data;
  } catch (error) {
    if (error instanceof Error) {
      throw new Error(`API Error: ${error.message}`);
    }
    throw new Error('Failed to connect to the backend API. Make sure it is running on http://localhost:8000');
  }
}
