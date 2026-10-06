import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# AI SOUND FOCUS - 50 CLASS MODEL TRAINING
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURE_FILE = os.path.join(
    BASE_DIR,
    "mfcc_features_50class.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "sound_classifier_50class.pkl"
)


print("=" * 60)
print("AI SOUND FOCUS - 50 CLASS MODEL TRAINING")
print("=" * 60)


# ------------------------------------------------------------
# 1. LOAD FEATURE DATA
# ------------------------------------------------------------

print("\nLoading feature dataset...")

df = pd.read_csv(FEATURE_FILE)

print(f"Total records : {len(df)}")


# ------------------------------------------------------------
# 2. SEPARATE FEATURES AND LABEL
# ------------------------------------------------------------

# filename and category are not input features
X = df.drop(columns=["filename", "category"])

# category is the target
y = df["category"]


print(f"Feature count : {X.shape[1]}")
print(f"Sound classes : {y.nunique()}")


# ------------------------------------------------------------
# 3. DISPLAY CLASSES
# ------------------------------------------------------------

classes = sorted(y.unique())

print("\nClasses used for training:")

for i, sound_class in enumerate(classes, start=1):
    print(f"{i:02d}. {sound_class}")


# ------------------------------------------------------------
# 4. TRAIN / TEST SPLIT
# ------------------------------------------------------------

print("\nCreating train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")


# ------------------------------------------------------------
# 5. CREATE RANDOM FOREST MODEL
# ------------------------------------------------------------

print("\nTraining Random Forest model...")
print("Please wait...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)


# ------------------------------------------------------------
# 6. TRAIN MODEL
# ------------------------------------------------------------

model.fit(X_train, y_train)

print("Model training completed!")


# ------------------------------------------------------------
# 7. PREDICTION
# ------------------------------------------------------------

print("\nEvaluating model...")

y_pred = model.predict(X_test)


# ------------------------------------------------------------
# 8. ACCURACY
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy : {accuracy * 100:.2f}%")


# ------------------------------------------------------------
# 9. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\nClassification Report:")
print("-" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ------------------------------------------------------------
# 10. SAVE MODEL
# ------------------------------------------------------------

with open(MODEL_FILE, "wb") as file:
    pickle.dump(model, file)


print("=" * 60)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print(f"\nModel file:")
print(MODEL_FILE)

print("\nModel details:")
print(f"Number of classes : {len(model.classes_)}")
print(f"Number of trees   : {model.n_estimators}")

print("\n50-class model training completed successfully! 🎧🤖")