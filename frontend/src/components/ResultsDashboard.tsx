import { PredictionResponse } from '../types';

interface Props {
  loading: boolean;
  prediction: PredictionResponse | null;
}

export function ResultsDashboard({ loading, prediction }: Props) {
  if (loading) {
    return <div className="text-white p-6 text-xl">Loading prediction...</div>;
  }

  // This keeps your debugging log and fixes the TypeScript "void" error
  if (!prediction) {
    console.log("Awaiting Prediction Data...");
    return null; 
  }

  // This lets you see the exact data coming back from Python
  console.log("Prediction Result Received:", prediction);

  // This prints ONLY the predicted class in giant text
  return (
    <div className="p-6 flex items-center justify-center h-full min-h-[400px]">
      <h1 className="text-6xl md:text-8xl font-black text-white tracking-widest uppercase text-center">
        {prediction.final_report.predicted_class}
      </h1>
    </div>
  );
}