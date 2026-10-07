
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "resumes.csv"
FEEDBACK_PATH = BASE_DIR / "data" / "feedback.csv"

def add_feedback_to_training_data():
    if not FEEDBACK_PATH.exists():
        print("No feedback file found. Nothing to retrain on.")
        return

    feedback = pd.read_csv(FEEDBACK_PATH)
    required = {"resume_text", "label"}
    if not required.issubset(feedback.columns):
        raise ValueError("feedback.csv must contain resume_text and label columns")

    existing = pd.read_csv(DATA_PATH)
    combined = pd.concat(
        [existing[["resume_text", "label"]], feedback[["resume_text", "label"]]],
        ignore_index=True
    ).drop_duplicates()

    combined.to_csv(DATA_PATH, index=False)
    print(f"Training dataset updated. Total records: {len(combined)}")

if __name__ == "__main__":
    add_feedback_to_training_data()
    from train import train
    train()
