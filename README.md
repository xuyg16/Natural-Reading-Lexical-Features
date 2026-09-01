
Lexical and language-model features for natural reading data, including word frequency and contextual surprisal.




# Natural Reading Lexical Features

Tools for extracting lexical and language-model-based predictors from natural reading text. Using data [ROAMM]((https://osf.io/kmvgb/overview)). 
[Data tutorials](https://data-brain-mind.github.io/tutorials/reading-observed-at-mindless-moments-roamm-a-simultaneous-eeg-and-eye-tracking-dataset-of-natural-reading-with-attention-annotations/)
This project is developed as part of a Master's thesis on continuous EEG decoding during natural reading. It provides word-level predictors that can later be aligned with eye-tracking fixations and EEG data.

## Features

Currently planned:

- **Word frequency**
  - lexical frequency / Zipf frequency
- **Word surprisal**
  - contextual surprisal estimated with a causal language model (e.g. GPT-2)

Additional lexical predictors may be added later.

## Workflow

```text
Natural reading text
        ↓
Recover word/token order
        ↓
Compute lexical features
    ├── word frequency
    └── LM surprisal
        ↓
Word-level feature table
        ↓
Merge with fixation / EEG events

