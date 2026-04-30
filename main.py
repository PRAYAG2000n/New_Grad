import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import CrossEncoder

app = FastAPI()
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

class Study(BaseModel):
    study_id: str
    study_description: str
    study_date: str

class CurrentStudy(Study):
    pass

class Case(BaseModel):
    case_id: str
    patient_id: str
    patient_name: str
    current_study: CurrentStudy
    prior_studies: List[Study]

class RequestPayload(BaseModel):
    challenge_id: str
    schema_version: int
    generated_at: str
    cases: List[Case]

class Prediction(BaseModel):
    case_id: str
    study_id: str
    predicted_is_relevant: bool

class ResponsePayload(BaseModel):
    predictions: List[Prediction]

@app.post("/predict", response_model=ResponsePayload)
async def predict(request: RequestPayload):
    predictions = []
    for case in request.cases:
        current_text = case.current_study.study_description
        for prior in case.prior_studies:
            prior_text = prior.study_description
            if not prior_text:
                predictions.append(Prediction(
                    case_id=case.case_id, 
                    study_id=prior.study_id, 
                    predicted_is_relevant=False
                ))
                continue
            
            # Cross-encoder takes a pair of texts and returns a relevance score (0-1)
            score = model.predict([(current_text, prior_text)])[0]
            # When score > 0.5, it's relevant
            is_relevant = score >= 0.5
            predictions.append(Prediction(
                case_id=case.case_id, 
                study_id=prior.study_id, 
                predicted_is_relevant=is_relevant
            ))
    return ResponsePayload(predictions=predictions)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
