import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer, util

app = FastAPI()
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
model = None
def load_model():
    global model
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    return model
cache = {}
def get_embedding(text):
    if text not in cache:
        cache[text] = load_model().encode(text, convert_to_tensor=True)
    return cache[text]

@app.post("/predict", response_model=ResponsePayload)
async def predict(request: RequestPayload):
    predictions = []
    for case in request.cases:
        current_text = case.current_study.study_description
        current_emb = get_embedding(current_text)
        for prior in case.prior_studies:
            if not prior.study_description:
                predictions.append(Prediction(case_id=case.case_id, study_id=prior.study_id, predicted_is_relevant=False))
                continue
            prior_emb = get_embedding(prior.study_description)
            similarity = util.pytorch_cos_sim(current_emb, prior_emb)[0][0].item()
            is_relevant = similarity >= 0.75
            predictions.append(Prediction(case_id=case.case_id, study_id=prior.study_id, predicted_is_relevant=is_relevant))
    return ResponsePayload(predictions=predictions)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
