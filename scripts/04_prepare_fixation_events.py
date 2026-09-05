from pathlib import Path
import logging

import numpy as np 
import pandas as pd 

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SYNCED_DIR = Path("/scratch/data/ROAMM/derivatives/synced")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "roamm_fixation_events.csv"

def nearest_sample_indices(time, onsets):
    """Return nearest EEG sample index for each fixation onset."""
    idx = np.searchsorted(time, onsets, side="left")
    idx = np.clip(idx, 1, len(time) - 1)

    left = idx - 1
    right = idx

    use_right = (np.abs(time[right] - onsets) < np.abs(time[left] - onsets))
    return np.where(use_right, right, left)

def extract_eye_events(df, eye, subject_id, run_num):
    tstart_col = f"fix_{eye}_tStart"
    tend_col = f"fix_{eye}_tEnd"
    duration_col = f"fix_{eye}_duration"
    word_col  = f"fix_{eye}_fixed_word"
    word_key_col = f"fix_{eye}_fixed_word_key"

    required = [
        "first_pass_reading",
        "page_num",
        "story_name",
        tstart_col,
        tend_col,
        duration_col,
        word_col,
        word_key_col
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {eye} eye: {missing}")
    
    # Only initial natural reading.
    mask = (df["first_pass_reading"].eq(True) & df[tstart_col].notna())
    events = df.loc[mask,["page_num", "story_name", tstart_col, tend_col, duration_col, word_col, word_key_col]].copy()

    # The same fixation can is repeated across many 256Hz samples.
    # Collapse it to one row per fixation event.
    events = (
        events
        .drop_duplicates(subset=[tstart_col])
        .sort_values(tstart_col)
        .reset_index(drop=True)
    )

    events = events.rename(
        columns={
            "page_num": "page",
            tstart_col: "onset_s",
            tend_col: "offset_s",
            duration_col: "duration_ms",
            word_col: "word",
            word_key_col: "word_key"
        }
    )

    events.insert(0, "subject_id", subject_id)
    events.insert(1, "run_num", run_num)
    events.insert(4, "eye", eye)

    return events

# Find locally available synced files.
all_files = sorted(SYNCED_DIR.glob("sub-*/*_mldata.pkl"))

# git-annex may show filenames whose actual content is not download.
# Path.exists() is False for those broken/unavailable symlinks.
files = [file for file in all_files if file.exists()]

if not files:
    raise FileNotFoundError(f"No locally available synced .pkl files found in {SYNCED_DIR}")

logger.info("Found %d locally available synced files", len(files))

all_events = []

# Process runs
for file in files:
    logger.info("Loading %s", file)

    df = pd.read_pickle(file)

    subject_id = file.parent.name

    run_values = df["run_num"].dropna().unique()

    if len(run_values) != 1:
        raise ValueError(f"Expected one unique run_num in {file}, found {run_values}")
    run_num = int(run_values[0])

    time = df["time"].to_numpy()

    if not np.all(np.diff(time) > 0):
        raise ValueError(f"EEG Time in {file} are not strictly increasing")

    for eye in ["L", "R"]:
        events = extract_eye_events(df, eye, subject_id, run_num)

        if events.empty:
            logger.warning("No fixation events found for %s eye in %s", subject_id, run_num, eye)
            continue

        onsets = events["onset_s"].to_numpy()

        sample_idx = nearest_sample_indices(time, onsets)
        events["eeg_sample"] = sample_idx
        # julia uses 1-based indexing
        events["latency"] = sample_idx + 1
        events["eeg_time_s"] = time[sample_idx]
        events["sync_error_ms"] = (events["eeg_time_s"] - events["onset_s"]) * 1000

        logger.info(
            "%s run %d eye %s: %d fixation, max sync error %.3f ms",
            subject_id,
            run_num,
            eye,
            len(events),
            events["sync_error_ms"].abs().max()
        )

        all_events.append(events)

# Combine + checks
events = pd.concat(all_events, ignore_index=True)
events = events[
    [
        "subject_id",
        "run_num",
        "story_name",
        "page",
        "eye",
        "onset_s",
        "offset_s",
        "duration_ms",
        "word",
        "word_key",
        "eeg_sample",
        "latency",
        "eeg_time_s",
        "sync_error_ms"
    ]
]

assert not events.duplicated(subset=["subject_id", "run_num", "eye", "onset_s"]).any()
assert (events["eeg_sample"] >= 0).all()
assert (events["latency"] >= 1).all()

# Save
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
events.to_csv(OUTPUT_PATH, index=False)
logger.info("Saved %d fixation events to %s", len(events), OUTPUT_PATH)