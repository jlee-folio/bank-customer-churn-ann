# Note: This explicitly excludes RowNumber, CustomerId, and Surname from the model's input features.

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "Churn_Modelling.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 10

TARGET = "Exited"

FEATURES = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]