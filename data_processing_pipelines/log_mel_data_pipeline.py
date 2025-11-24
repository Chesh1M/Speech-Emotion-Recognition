import os
import numpy as np
import pandas as pd
import librosa
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import time

# ----------------------------
# PARAMETERS
# ----------------------------
CREMA = '../data_cleaning_and_eda/crema_d_labels.csv'
RAVDESS = '../data_cleaning_and_eda/ravdess_labels.csv'
TESS = '../data_cleaning_and_eda/tess_labels.csv'
SR = 44100
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
WINDOW_TYPE = "hann"
SAVE_TO_DISK = True
TEST_SIZE = 0.2
VAL_SIZE = 0.2
EXTRACT_TEST_SET = True
SEED = 42
OUTPUT_DIR = "../datasets_log_mel_combined"

os.makedirs(OUTPUT_DIR, exist_ok=True)

start_time = time.time()

# ----------------------------
# LOAD DATASETS
# ----------------------------
crema_data = pd.read_csv(CREMA)
ravdess_data = pd.read_csv(RAVDESS)
tess_data = pd.read_csv(TESS)
df = pd.concat([ravdess_data, crema_data, tess_data], join='inner')

# ----------------------------
# TRAIN/VAL/TEST SPLIT
# ----------------------------
all_files = df["file_path"].values

train_val_files, test_files = train_test_split(
    all_files, test_size=TEST_SIZE, random_state=SEED,
    stratify=df["Emotion"].values
)
train_files, val_files = train_test_split(
    train_val_files, test_size=VAL_SIZE/(1-TEST_SIZE), random_state=SEED,
    stratify=df.set_index("file_path").loc[train_val_files]["Emotion"].values
)

splits = {
    "train": train_files,
    "val": val_files,
    "test": test_files
}

print(f"Split sizes: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

# ----------------------------
# FEATURE EXTRACTION (LOG-MEL)
# ----------------------------
def extract_features(file_list):
    mel_sequences = []
    labels = []

    for file in file_list:
        try:
            # Load audio
            y, sr = librosa.load(file, sr=SR, res_type='kaiser_fast')

            # Compute log-mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS,
                window=WINDOW_TYPE
            )
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

            # Transpose → (time, mel_bins) for deep learning input
            mel_sequences.append(log_mel_spec.T)

            # Emotion label
            emotion = df.set_index("file_path").loc[file, "Emotion"]
            labels.append(emotion)

        except Exception as e:
            print(f"⚠️ Error processing {file}: {e}")

    # Pad all sequences to equal length
    X_deep = pad_sequences(mel_sequences, padding='post', dtype='float32')
    y_deep = np.array(labels)

    return X_deep, y_deep

# ----------------------------
# EXPORT FUNCTIONS
# ----------------------------
def export_deep_to_npy(X, y, split_name):
    X_path = os.path.join(OUTPUT_DIR, f"X_logmel_{split_name}.npy")
    y_path = os.path.join(OUTPUT_DIR, f"y_logmel_{split_name}.npy")
    np.save(X_path, X)
    np.save(y_path, y)
    print(f"✅ Saved: X_logmel_{split_name}.npy ({X.shape}), y_logmel_{split_name}.npy ({y.shape})")

# ----------------------------
# MAIN LOOP
# ----------------------------
for split_name, files in splits.items():
    feature_start = time.time()
    print(f"\n🔸 Extracting log-mel features for {split_name} set...")

    X_deep, y_deep = extract_features(files)
    export_deep_to_npy(X_deep, y_deep, split_name)

    print(f"⏱️ {split_name} extraction completed in {time.time() - feature_start:.2f}s")

# ----------------------------
# PRINT PIPELINE TIME ELAPSED
# ----------------------------
elapsed = time.time() - start_time
print(f"\nPipeline completed in {elapsed:.2f} seconds")
