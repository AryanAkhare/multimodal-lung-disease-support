from pydantic import BaseModel
from typing import List

class BranchScore(BaseModel):
    label: str
    confidence: float
    message: str

class FinalReport(BaseModel):
    primary_finding: str
    overall_confidence: float
    recommendation: List[str]
    note: str

class PredictionResponse(BaseModel):
    image_branch: BranchScore
    audio_branch: BranchScore
    tabular_branch: BranchScore
    final_report: FinalReport
