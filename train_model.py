import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------
# Load feature dataset
# --------------------------------

data = pd.read_csv("sound_features.csv")

print("\n✅ Feature dataset loaded!")
print("Total records:", len(data))


# --------------------------------
# Select features
# --------------------------------

features = [
    "rms",
    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff"
]

X = data[features]
y = data["category"]


# --------------------------------
# Split dataset
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------
# Create Random Forest model
# --------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# --------------------------------
# Train model
# --------------------------------

print("\n🤖 Training Random Forest model...")

model.fit(X_train, y_train)


# --------------------------------
# Test model
# --------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n" + "=" * 50)
print("🤖 MODEL TRAINING COMPLETED")
print("=" * 50)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")


# --------------------------------
# Classification report
# --------------------------------

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions
    )
)


# --------------------------------
# Save trained model
# --------------------------------

with open("sound_classifier.pkl", "wb") as file:
    pickle.dump(model, file)


print("\n✅ Model saved successfully!")

print(
    "Saved as: sound_classifier.pkl"
)