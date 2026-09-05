from pathlib import Path
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVENTS_PATH = PROJECT_ROOT / "outputs" / "roamm_fixation_events.csv"
LEXICAL_PATH = PROJECT_ROOT / "outputs" / "roamm_words_with_surprisal.csv"
FULL_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "roamm_b2b_events_full.csv"
ANALYSIS_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "roamm_b2b_events.csv"

# Load
logger.info("Loading fixation events from %s", EVENTS_PATH)
events = pd.read_csv(EVENTS_PATH, keep_default_na=False)

logger.info("Loading lexical features from %s", LEXICAL_PATH)
lexical = pd.read_csv(LEXICAL_PATH, keep_default_na=False)

logger.info("Loaded %d fixation events", len(events))
logger.info("Loaded %d lexical word occurrences", len(lexical))

# Restore numeric missing values after kepp_default_na=False
for col in [
    "frequency_zipf",
    "surprisal_bits",
    "word_length"
]:
    if col in lexical.columns:
        lexical[col] = pd.to_numeric(lexical[col], errors="coerce")


# Keep only fixations mapped to a word
n_unmatched = (events["word_key"] == "").sum()

logger.info("%d fixation events have no mapped word", n_unmatched)

events_word = events[events["word_key"] != ""].copy()

# Prepare lexical columns
lexical_features = lexical[
    [
        "word_key",
        "words",
        "word_position",
        "sentence_id",
        "lookup_word",
        "word_length",
        "frequency_zipf",
        "surprisal_bits",
        "surprisal_model",
    ]
].rename(
    columns={
        "words": "text_word"
    }
)

# Merge
merged = events_word.merge(
    lexical_features,
    on="word_key",
    how="left",
    validate="many_to_one",
    indicator=True
)

# Checks 
merge_counts = merged["_merge"].value_counts()

logger.info("Merge results:\n%s", merge_counts.to_string())

n_failed = (merged["_merge"] != "both").sum()

if n_failed:
    raise ValueError(
        f"{n_failed} word-linked fixations failed lexical merge."
    )
    #logger.warning("%d word-linked fixations failed lexical merge", n_failed)

# word_key should identify the same word in both tables
matched = merged["_merge"] == "both"

n_word_mismatch = (merged.loc[matched, "word"] != merged.loc[matched, "text_word"]).sum()

if n_word_mismatch:
    logger.warning("%d merged events have mismatching word labels", n_word_mismatch)

merged["is_lexical"] = (
    (merged["_merge"] == "both")
    & merged["lookup_word"].notna()
    & (merged["lookup_word"] != "")
)
merged = merged.drop(columns=["_merge"])

# Save
full_events = merged.copy()
FULL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
full_events.to_csv(FULL_OUTPUT_PATH, index=False)
logger.info("Saved %d full events to %s", len(full_events), FULL_OUTPUT_PATH)


analysis_columns = [
    "subject_id",
    "run_num",
    "story_name",
    "page",
    "eye",
    "onset_s",
    "duration_ms",
    "latency",
    "word",
    "word_key",
    "word_position",
    "sentence_id",
    "word_length",
    "frequency_zipf",
    "surprisal_bits"
]

analysis_events = merged.loc[merged["is_lexical"], analysis_columns].copy()
analysis_events.to_csv(ANALYSIS_OUTPUT_PATH, index=False)
logger.info("Saved %d B2B events to %s", len(analysis_events), ANALYSIS_OUTPUT_PATH)