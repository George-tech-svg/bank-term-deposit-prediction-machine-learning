import joblib

print("Loading model...")
model = joblib.load("best_model.pkl")
print("Model loaded successfully")

print("Loading preprocessor...")
preprocessor = joblib.load("preprocessor.pkl")
print("Preprocessor loaded successfully")

print("Files are ready for deployment.")