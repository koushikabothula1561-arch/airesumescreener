# AI-Based Resume Screening System with MLOps

An end-to-end beginner-friendly MLOps project that automatically classifies resumes using NLP and exposes predictions through FastAPI.

## Architecture

Resume text → TF-IDF NLP features → Logistic Regression → Resume category  
                                     ↓  
                                   MLflow tracking  
                                     ↓  
                             Saved production model  
                                     ↓  
                                 FastAPI REST API

## Features

- NLP-based resume classification
- TF-IDF with unigrams and bigrams
- Logistic Regression classifier
- Train/test evaluation
- MLflow experiment and metric tracking
- Saved model with Joblib
- FastAPI prediction endpoint
- TXT resume upload endpoint
- Feedback-based retraining pipeline
- VS Code configuration
- Sample dataset for immediate testing

## Resume Categories

The included demo dataset has:
- Software Engineer
- Data Scientist
- HR
- Network Engineer
- Marketing
- DevOps

This is a demonstration project, not a production hiring decision system. Real HR systems should include privacy protection, bias testing, human review, explainability, and legal/compliance checks.

## 1. Open in VS Code

Open this folder:

```text
AI_Resume_Screening_MLOps
```

## 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Train the model

```bash
python src/train.py
```

The model will be saved as:

```text
models/resume_classifier.joblib
```

MLflow data will be stored in:

```text
mlruns/
```

## 5. Start MLflow UI

Open a second terminal with the virtual environment activated:

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Open the displayed local MLflow address in your browser.

## 6. Start FastAPI

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

The Swagger UI lets you test the API.

## API Example

POST `/predict`

```json
{
  "text": "Python developer with FastAPI, Docker, AWS, SQL and machine learning experience."
}
```

Example response:

```json
{
  "prediction": "Software Engineer",
  "confidence": 0.72,
  "scores": {
    "Data Scientist": 0.08,
    "DevOps": 0.10,
    "HR": 0.02,
    "Marketing": 0.01,
    "Network Engineer": 0.03,
    "Software Engineer": 0.76
  }
}
```

The exact scores depend on the trained model.

## Upload a TXT Resume

Use POST `/predict-file` in Swagger UI and upload a `.txt` resume.

For PDF/DOCX support, extend `src/extract_text.py` with libraries such as PyMuPDF and python-docx.

## Continuous Improvement / Retraining

Add corrected examples to:

```text
data/feedback.csv
```

Then run:

```bash
python src/retrain.py
```

The retraining script:
1. Reads new labeled feedback.
2. Merges it with the original training dataset.
3. Removes duplicates.
4. Retrains the NLP model.
5. Logs the new run and metrics to MLflow.
6. Replaces the saved model.

## Suggested GitHub Project Structure

```text
AI_Resume_Screening_MLOps/
├── app/
│   └── main.py
├── data/
│   ├── resumes.csv
│   └── feedback.csv
├── models/
├── reports/
├── mlruns/
├── src/
│   ├── extract_text.py
│   ├── retrain.py
│   └── train.py
├── .vscode/
│   ├── settings.json
│   └── launch.json
├── .gitignore
├── requirements.txt
└── README.md
```

## Future Improvements

- PDF and DOCX resume parsing
- Named Entity Recognition for skills and education
- Skill extraction and matching against job descriptions
- Candidate ranking
- Explainable predictions
- PostgreSQL database
- Docker deployment
- GitHub Actions CI/CD
- MLflow Model Registry
- Automated scheduled retraining
- Bias/fairness evaluation
- Authentication and role-based access

## Resume Screening Flow

1. HR uploads a resume.
2. API extracts/receives resume text.
3. NLP converts text into TF-IDF features.
4. ML model predicts the most relevant job category.
5. API returns prediction and confidence scores.
6. Corrected HR labels can be stored as feedback.
7. Retraining incorporates new labeled examples.
8. MLflow records each training run and its metrics.
