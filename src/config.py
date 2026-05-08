# Configuration settings for the bank term deposit prediction project

# File paths - using relative paths from the scripts folder
RAW_DATA_PATH = "../data/bank-full.csv"
CLEANED_DATA_PATH = "../outputs/cleaned_data/bank_cleaned.csv"
MODEL_SAVE_PATH = "../outputs/models/best_model.pkl"
METRICS_SAVE_PATH = "../outputs/models/model_metrics.json"

# Data column names
TARGET_COLUMN = "y"

# Categorical columns that need encoding
CATEGORICAL_COLUMNS = [
    "job", "marital", "education", "default", "housing", 
    "loan", "contact", "month", "poutcome"
]

# Numerical columns
NUMERICAL_COLUMNS = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

# Columns to drop
COLUMNS_TO_DROP = []

# Test size for train-test split
TEST_SIZE = 0.2

# Random seed for reproducibility
RANDOM_SEED = 42

# Model parameters for Random Forest
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": RANDOM_SEED
}