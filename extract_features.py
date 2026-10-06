import pandas as pd
import numpy as np
from pathlib import Path
from scipy.io import wavfile


# -----------------------------
# Project paths
# -----------------------------

project_folder = Path(__file__).parent

audio_folder = (
    project_folder
    / "ESC-50-master"
    / "ESC-50-master"
    / "audio"
)

metadata_file = (
    project_folder
    / "ESC-50-master"
    / "ESC-50-master"
    / "meta"
    / "esc50.csv"
)


# -----------------------------
# Read metadata
# -----------------------------

df = pd.read_csv(metadata_file)


# -----------------------------
# Select focus-related classes
# -----------------------------

focus_classes = [
    "rain",
    "sea_waves",
    "wind",
    "keyboard_typing",
    "engine",
    "car_horn",
    "siren",
    "thunderstorm"
]

df = df[df["category"].isin(focus_classes)].copy()


# -----------------------------
# Feature extraction function
# -----------------------------

def extract_features(file_path):

    sample_rate, audio = wavfile.read(file_path)

    # Convert stereo to mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Convert to float
    audio = audio.astype(np.float64)

    # Normalize
    max_value = np.max(np.abs(audio))

    if max_value > 0:
        audio = audio / max_value

    # -------------------------
    # RMS Energy
    # -------------------------

    rms = np.sqrt(np.mean(audio ** 2))

    # -------------------------
    # Zero Crossing Rate
    # -------------------------

    zero_crossings = np.sum(
        np.abs(np.diff(np.sign(audio)))
    ) / len(audio)

    # -------------------------
    # FFT
    # -------------------------

    fft_values = np.abs(np.fft.rfft(audio))

    frequencies = np.fft.rfftfreq(
        len(audio),
        d=1 / sample_rate
    )

    # Avoid division by zero
    total_energy = np.sum(fft_values)

    if total_energy == 0:
        spectral_centroid = 0
        spectral_bandwidth = 0
        spectral_rolloff = 0

    else:

        # -------------------------
        # Spectral Centroid
        # -------------------------

        spectral_centroid = (
            np.sum(frequencies * fft_values)
            / total_energy
        )

        # -------------------------
        # Spectral Bandwidth
        # -------------------------

        spectral_bandwidth = np.sqrt(
            np.sum(
                ((frequencies - spectral_centroid) ** 2)
                * fft_values
            )
            / total_energy
        )

        # -------------------------
        # Spectral Rolloff
        # -------------------------

        cumulative_energy = np.cumsum(fft_values)

        rolloff_threshold = 0.85 * cumulative_energy[-1]

        rolloff_index = np.where(
            cumulative_energy >= rolloff_threshold
        )[0]

        if len(rolloff_index) > 0:
            spectral_rolloff = frequencies[rolloff_index[0]]
        else:
            spectral_rolloff = 0

    return [
        rms,
        zero_crossings,
        spectral_centroid,
        spectral_bandwidth,
        spectral_rolloff
    ]


# -----------------------------
# Process all audio files
# -----------------------------

results = []

print("\n🎧 Extracting audio features...\n")

for index, row in df.iterrows():

    file_path = audio_folder / row["filename"]

    if not file_path.exists():
        print("⚠️ File not found:", row["filename"])
        continue

    features = extract_features(file_path)

    results.append([
        row["filename"],
        row["category"],
        features[0],
        features[1],
        features[2],
        features[3],
        features[4]
    ])

    print(
        f"Processed: {row['filename']} → {row['category']}"
    )


# -----------------------------
# Create feature dataset
# -----------------------------

feature_df = pd.DataFrame(
    results,
    columns=[
        "filename",
        "category",
        "rms",
        "zero_crossing_rate",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff"
    ]
)


# -----------------------------
# Save CSV
# -----------------------------

output_file = project_folder / "sound_features.csv"

feature_df.to_csv(
    output_file,
    index=False
)


print("\n" + "=" * 50)
print("✅ FEATURE EXTRACTION COMPLETED!")
print("=" * 50)

print("\nTotal processed files:", len(feature_df))

print("\nClasses:")
print(feature_df["category"].value_counts())

print("\nFeature dataset saved at:")
print(output_file)