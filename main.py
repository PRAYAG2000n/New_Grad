import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import CrossEncoder
import os

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

score_cache = {}

@app.post("/predict", response_model=ResponsePayload)
async def predict(request: RequestPayload):
    predictions = []

    pairs = []
    pair_meta = [] 

    for case in request.cases:
        current_text = case.current_study.study_description or ""
        for prior in case.prior_studies:
            prior_text = prior.study_description or ""
            if not prior_text or not current_text:
                pair_meta.append((case.case_id, prior.study_id, None))
                continue
            cache_key = (current_text, prior_text)
            if cache_key in score_cache:
                pair_meta.append((case.case_id, prior.study_id, score_cache[cache_key]))
            else:
                pairs.append((current_text, prior_text))
                pair_meta.append((case.case_id, prior.study_id, "PENDING"))

    if pairs:
        scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
        score_iter = iter(scores)
        for i, (cid, sid, val) in enumerate(pair_meta):
            if val == "PENDING":
                s = float(next(score_iter))
                score_cache[(request.cases[0].current_study.study_description, "")] = s 
                pair_meta[i] = (cid, sid, s)


    # Build predictions
    for cid, sid, score in pair_meta:
        if score is None:
            predictions.append(Prediction(case_id=cid, study_id=sid, predicted_is_relevant=False))
        else:
            is_relevant = bool(score >= 0.0)
            predictions.append(Prediction(case_id=cid, study_id=sid, predicted_is_relevant=is_relevant))

    return ResponsePayload(predictions=predictions)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"status": "ok", "endpoint": "/predict"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
