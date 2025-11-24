## ==============================================================================================
## This file only combines the 3 audio data corpora & splits them into separate train & test sets, so that augmentation can be done within each K-fold to prevent data leakage.==============================================================================================

import os
import pandas as pd
from sklearn.model_selection import train_test_split

CREMA = '../data_cleaning_and_eda/crema_d_labels.csv'
RAVDESS = '../data_cleaning_and_eda/ravdess_labels.csv'
TESS = '../data_cleaning_and_eda/tess_labels.csv'
OUTPUT_DIR = "../datasets_combined_augmented_CV"
SEED = 42

# Create output folder if doesn't already exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load datasets
crema = pd.read_csv(CREMA)
ravdess = pd.read_csv(RAVDESS)
tess = pd.read_csv(TESS)

df = pd.concat([crema, ravdess, tess], ignore_index=True)

# Split only train/test
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["Emotion"], random_state=SEED
)

# Save splits (raw paths only)
train_df.to_csv(f"{OUTPUT_DIR}/train_split.csv", index=False)
test_df.to_csv(f"{OUTPUT_DIR}/test_split.csv", index=False)

print(train_df.shape, test_df.shape)