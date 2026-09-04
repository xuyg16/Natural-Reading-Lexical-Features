from pathlib import Path 
import pandas as pd 

DATA_DIR = Path("/scratch/data/ROAMM/text")
OUTPUT_PATH = Path("outputs/roamm_words.csv")

files = sorted(DATA_DIR.glob("*_coordinates.csv"))

tables = []

for file in files:
    print("Loading:", file.name)

    df = pd.read_csv(file)

    # story name from filename
    story_name = file.stem.replace("_coordinates", "")
    df["story_name"] = story_name

    # original order within the article
    df = df.reset_index(drop=True)
    df["word_position"] = df.index 
    tables.append(df)

words = pd.concat(tables, ignore_index=True)

# keep only the columns we currently need
wrods = words[
    [
        "story_name",
        "page",
        "sentence_id",
        "sentence",
        "word_position",
        "words",
        "word_key",
    ]
]

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
words.to_csv(OUTPUT_PATH, index=False)

print("\nShape:", words.shape)
print("\nStories:")
print(words["story_name"].value_counts())

print("\nFirst rows:")
print(words.head(30))

print(f"\nSaved to: {OUTPUT_PATH}")