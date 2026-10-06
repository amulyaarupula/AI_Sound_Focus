import pandas as pd
from pathlib import Path

project_folder = Path(__file__).parent

metadata_file = (
    project_folder
    / "ESC-50-master"
    / "ESC-50-master"
    / "meta"
    / "esc50.csv"
)

df = pd.read_csv(metadata_file)

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

filtered_df = df[df["category"].isin(focus_classes)]

print("\n🎧 Focus-related sound classes\n")

for sound_class in focus_classes:
    count = len(filtered_df[filtered_df["category"] == sound_class])
    print(f"{sound_class:20} → {count} audio files")

print("\nTotal selected audio files:", len(filtered_df))