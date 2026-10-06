import pandas as pd
from pathlib import Path

# Project folder
project_folder = Path(__file__).parent

# ESC-50 metadata file
metadata_file = (
    project_folder
    / "ESC-50-master"
    / "ESC-50-master"
    / "meta"
    / "esc50.csv"
)

# Check if file exists
if not metadata_file.exists():
    print("❌ Metadata file not found!")
    print("Expected location:")
    print(metadata_file)
    exit()

# Read metadata
df = pd.read_csv(metadata_file)

print("\n✅ ESC-50 metadata loaded successfully!\n")

print("Columns:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

print("\nTotal audio records:")
print(len(df))

print("\nTotal sound classes:")
print(df["category"].nunique())

print("\nSound classes:")
print(df["category"].unique())