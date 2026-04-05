import { TabularInputs, PredictionResponse } from '../types';

const API_BASE_URL = 'http://localhost:8000';

export const checkApiHealth = async (): Promise<boolean> => {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.ok;
  } catch (error) {
    return false;
  }
};

export const predictPatientData = async ({
  imageFile,
  audioFile,
  tabular,
}: {
  imageFile: File | null;
  audioFile: File | null;
  tabular: TabularInputs;
}): Promise<PredictionResponse> => {
  const formData = new FormData();

  if (imageFile) {
    formData.append('image', imageFile);
  }

  if (audioFile) {
    formData.append('audio', audioFile);
  }

  (Object.keys(tabular) as Array<keyof TabularInputs>).forEach((key) => {
    formData.append(key, String(tabular[key]));
  });

  const response = await fetch(`${API_BASE_URL}/api/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || 'Prediction failed');
  }

  return response.json();
};