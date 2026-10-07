
from pathlib import Path
import joblib
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "resume_classifier.joblib"
HTML_PATH = BASE_DIR / "app" / "index.html"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="AI Resume Screening System",
    version="1.0.0",
    description="NLP + MLflow + FastAPI resume screening API"
)


class ResumeText(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
def root():
    return HTML_PATH.read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "TF-IDF + Logistic Regression"
    }


@app.post("/predict")
def predict(resume: ResumeText):
    prediction = model.predict([resume.text])[0]

    probabilities = model.predict_proba([resume.text])[0]
    classes = model.classes_

    scores = {
        str(label): round(float(score), 4)
        for label, score in zip(classes, probabilities)
    }

    confidence = round(float(max(probabilities)), 4)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "scores": scores
    }


@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")

    prediction = model.predict([text])[0]

    probabilities = model.predict_proba([text])[0]
    classes = model.classes_

    scores = {
        str(label): round(float(score), 4)
        for label, score in zip(classes, probabilities)
    }

    confidence = round(float(max(probabilities)), 4)

    return {
        "filename": file.filename,
        "prediction": prediction,
        "confidence": confidence,
        "scores": scores
    }
