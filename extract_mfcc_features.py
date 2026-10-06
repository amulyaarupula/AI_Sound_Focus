import os
import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm

# ==============================
# PROJECT PATHS
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AUDIO_DIR = os.path.join(
    BASE_DIR,
    "ESC-50-master",
    "ESC-50-master",
    "audio"
)

META_FILE = os.path.join(
    BASE_DIR,
    "ESC-50-master",
    "ESC-50-master",
    "meta",
    "esc50.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "mfcc_features_50class.csv"
)


# ==============================
# CHECK DATASET
# ==============================

if not os.path.exists(AUDIO_DIR):
    print("ERROR: Audio folder not found.")
    print("Expected location:")
    print(AUDIO_DIR)
    raise SystemExit

if not os.path.exists(META_FILE):
    print("ERROR: esc50.csv not found.")
    print("Expected location:")
    print(META_FILE)
    raise SystemExit


# ==============================
# LOAD DATASET METADATA
# ==============================

metadata = pd.read_csv(META_FILE)

print("=" * 60)
print("AI SOUND FOCUS - FEATURE EXTRACTION")
print("=" * 60)

print(f"Total audio files in metadata : {len(metadata)}")
print(f"Total sound classes           : {metadata['category'].nunique()}")

print("\nStarting MFCC feature extraction...")
print("This may take some time.\n")


# ==============================
# FEATURE EXTRACTION FUNCTION
# ==============================

def extract_features(file_path):

    audio, sample_rate = librosa.load(
        file_path,
        sr=None,
        mono=True
    )

    features = {}

    # --------------------------------
    # MFCC FEATURES
    # --------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=13
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    for i in range(13):
        features[f"mfcc_mean_{i + 1}"] = mfcc_mean[i]

    for i in range(13):
        features[f"mfcc_std_{i + 1}"] = mfcc_std[i]

    # --------------------------------
    # RMS ENERGY
    # --------------------------------

    rms = librosa.feature.rms(y=audio)

    features["rms"] = np.mean(rms)

    # --------------------------------
    # ZERO CROSSING RATE
    # --------------------------------

    zcr = librosa.feature.zero_crossing_rate(y=audio)

    features["zero_crossing_rate"] = np.mean(zcr)

    # --------------------------------
    # SPECTRAL CENTROID
    # --------------------------------

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate
    )

    features["spectral_centroid"] = np.mean(
        spectral_centroid
    )

    # --------------------------------
    # SPECTRAL BANDWIDTH
    # --------------------------------

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sample_rate
    )

    features["spectral_bandwidth"] = np.mean(
        spectral_bandwidth
    )

    return features


# ==============================
# PROCESS ALL AUDIO FILES
# ==============================

records = []

successful = 0
failed = 0

for _, row in tqdm(
    metadata.iterrows(),
    total=len(metadata),
    desc="Extracting features"
):

    filename = row["filename"]
    category = row["category"]

    file_path = os.path.join(
        AUDIO_DIR,
        filename
    )

    try:

        features = extract_features(file_path)

        # Add dataset information
        features["filename"] = filename
        features["category"] = category

        records.append(features)

        successful += 1

    except Exception as e:

        failed += 1

        print(
            f"\nError processing {filename}: {e}"
        )


# ==============================
# CREATE DATAFRAME
# ==============================

if len(records) == 0:

    print("\nERROR: No features were extracted.")
    raise SystemExit


df = pd.DataFrame(records)


# ==============================
# ORGANIZE COLUMNS
# ==============================

feature_columns = [
    column
    for column in df.columns
    if column not in ["filename", "category"]
]

df = df[
    ["filename", "category"] + feature_columns
]


# ==============================
# SAVE FEATURES
# ==============================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==============================
# FINAL REPORT
# ==============================

print("\n" + "=" * 60)
print("FEATURE EXTRACTION COMPLETED")
print("=" * 60)

print(f"Successful files : {successful}")
print(f"Failed files     : {failed}")

print(
    f"Feature count    : {len(feature_columns)}"
)

print(
    f"Sound classes    : {df['category'].nunique()}"
)

print(
    f"Output file      : {OUTPUT_FILE}"
)

print("\nClasses found:")

for index, category in enumerate(
    sorted(df["category"].unique()),
    start=1
):

    print(
        f"{index:02d}. {category}"
    )

print("\nFeature extraction is ready for model training.")