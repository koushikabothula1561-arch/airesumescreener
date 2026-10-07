
from pathlib import Path
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "resumes.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

def train():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["resume_text", "label"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["resume_text"],
        df["label"],
        test_size=0.30,
        random_state=42,
        stratify=df["label"]
    )

    max_features = 5000
    C = 2.0

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=max_features
        )),
        ("classifier", LogisticRegression(
            C=C,
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    mlflow.set_experiment("AI_Resume_Screening")

    with mlflow.start_run(run_name="resume-classifier"):
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="weighted")

        mlflow.log_param("model", "TF-IDF + Logistic Regression")
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("C", C)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("weighted_f1", f1)

        report = classification_report(y_test, predictions)
        report_path = BASE_DIR / "reports" / "classification_report.txt"
        report_path.parent.mkdir(exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        mlflow.log_artifact(str(report_path))

        mlflow.sklearn.log_model(pipeline, "resume_model")

        model_path = MODEL_DIR / "resume_classifier.joblib"
        joblib.dump(pipeline, model_path)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Weighted F1: {f1:.4f}")
        print(f"Model saved to: {model_path}")
        print(report)

if __name__ == "__main__":
    train()