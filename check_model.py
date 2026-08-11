from pathlib import Path
import joblib

model_dir = Path(__file__).resolve().parent / "saved_models"
metadata = joblib.load(model_dir / "metadata.pkl")

print("Features:", metadata["features"])
print("Training records:", metadata["training_records"])
print("Validation records:", metadata["validation_records"])
print("Best validation loss:", metadata["best_validation_loss"])

print("\nThresholds")
print("Warning:", metadata["warning_threshold"])
print("Critical:", metadata["critical_threshold"])
print("Failure:", metadata["failure_threshold"])
