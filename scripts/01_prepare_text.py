from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("/scratch/data/ROAMM/text")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "roamm_words.csv"


files = sorted(DATA_DIR.glob("*_coordinates.csv"))

if not files:
    raise FileNotFoundError(f"No coordinate CSV files found in {DATA_DIR}")


logger.info("Found %d coordinate files", len(files))

tables = []

for file in files:
    logger.info("Loading %s", file.name)
    df = pd.read_csv(file, keep_default_na=False)

    story_name = file.stem.replace("_coordinates", "")
    df["story_name"] = story_name

    df = df.reset_index(drop=True)
    df["word_position"] = df.index

    tables.append(df)


words = pd.concat(tables, ignore_index=True)

words = words[
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


# Minimal checks
assert words["word_key"].notna().all()
assert not words["word_key"].duplicated().any()
assert not words.duplicated(
    subset=["story_name", "word_position"]
).any()


OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
words.to_csv(OUTPUT_PATH, index=False)

logger.info("Saved %d words to %s", len(words), OUTPUT_PATH)