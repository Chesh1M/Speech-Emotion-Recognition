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
CREMA = 'crema_d_labels.csv'
RAVDESS = 'ravdess_labels.csv'
TESS = 'tess_labels.csv'
SR = 44100
N_MFCC = 30
N_FFT = 2048
HOP_LENGTH = 512
WINDOW_TYPE = "hann"
SAVE_TO_DISK = True
TEST_SIZE = 0.2
VAL_SIZE = 0.2
EXTRACT_TEST_SET = True
SEED = 42
OUTPUT_DIR = "datasets_combined_augmented"
AUGMENT = True  # Toggle augmentation on/off

# Create output folder if doesn't already exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Start time capture
start_time = time.time()

# ----------------------------
# LOAD DATASETS
# ----------------------------
crema_data = pd.read_csv(CREMA)
ravdess_data = pd.read_csv(RAVDESS)
tess_data = pd.read_csv(TESS)
df = pd.concat([ravdess_data, crema_data, tess_data], join='inner')

# ----------------------------
# TRAIN/VAL/TEST SPLIT (file-level)
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

print(f"Split sizes: train={len(train_files)}, test={len(test_files)}")

# ----------------------------
# AUGMENTATION FUNCTIONS
# ----------------------------
def add_noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    return data + noise_amp * np.random.normal(size=data.shape[0])

def pitch_shift(data, sr, pitch_factor=0.7):
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=pitch_factor)

def stretch(data, rate=0.9):
    data = np.asarray(data, dtype=np.float32).flatten()  # ensure 1D
    return librosa.effects.time_stretch(data, rate=rate)

# ----------------------------
# FEATURE EXTRACTION FUNCTION
# ----------------------------
def extract_features(file_list):
    classical_features = []
    sequence_features = []
    labels = []

    for file in file_list:
        # Load audio
        y, sr = librosa.load(file, sr=SR, res_type='kaiser_fast')

        versions = [y]  # original
        if AUGMENT:
            versions.append(add_noise(y))
            versions.append(pitch_shift(y, sr))
            versions.append(stretch(y, rate=0.9))

        for audio in versions:
            # === MFCCs ===
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=N_MFCC,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                win_length=N_FFT,
                window=WINDOW_TYPE
            )

            # === Delta and Delta-Delta ===
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)

            # === ZCR ===
            zcr = librosa.feature.zero_crossing_rate(
                audio,
                frame_length=N_FFT,
                hop_length=HOP_LENGTH
            )

            # === RMSE ===
            rmse = librosa.feature.rms(
                y=audio,
                frame_length=N_FFT,
                hop_length=HOP_LENGTH
            )

            # === Stack features ===
            features_combined = np.vstack([mfcc, delta, delta2, zcr, rmse])

            # Classical ML: average across time
            features_mean = np.mean(features_combined, axis=1)
            classical_features.append(features_mean)

            # Deep Learning: keep full sequence
            sequence_features.append(features_combined.T)

            # Label
            emotion = df.set_index("file_path").loc[file, "Emotion"]
            labels.append(emotion)

    # Convert to arrays
    X_classical = np.array(classical_features)
    y_classical = np.array(labels)

    X_deep = pad_sequences(sequence_features, padding='post', dtype='float32')
    y_deep = np.array(labels)

    return X_classical, y_classical, X_deep, y_deep

# ----------------------------
# EXPORT FUNCTIONS
# ----------------------------
def export_classical_to_csv(X, y, split_name):
    feature_names = (
        [f"mfcc_{i+1}" for i in range(N_MFCC)] +
        [f"delta_{i+1}" for i in range(N_MFCC)] +
        [f"delta2_{i+1}" for i in range(N_MFCC)] +
        ["zcr", "rmse"]
    )
    df_out = pd.DataFrame(X, columns=feature_names)
    df_out["Emotion"] = y
    csv_path = os.path.join(OUTPUT_DIR, f"classical_{split_name}.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"✅ Saved classical features: classical_{split_name}.csv — shape {df_out.shape}")

def export_deep_to_npy(X, y, split_name):
    X_path = os.path.join(OUTPUT_DIR, f"X_deep_{split_name}.npy")
    y_path = os.path.join(OUTPUT_DIR, f"y_deep_{split_name}.npy")
    np.save(X_path, X)
    np.save(y_path, y)
    print(f"✅ Saved deep features: X_deep_{split_name}.npy {X.shape}, y_deep_{split_name}.npy {y.shape}")

# ----------------------------
# MAIN EXTRACTION LOOP
# ----------------------------
for split_name, files in splits.items():
    feature_start = time.time()
    print(f"\n🔸 Extracting features for {split_name} set...")

    X_classical, y_classical, X_deep, y_deep = extract_features(files)

    export_classical_to_csv(X_classical, y_classical, split_name)
    export_deep_to_npy(X_deep, y_deep, split_name)

    print(f"⏱️ {split_name} extraction completed in {time.time() - feature_start:.2f}s")

# ----------------------------
# PRINT PIPELINE TIME ELAPSED
# ----------------------------
end_time = time.time()
elapsed = end_time - start_time
print(f"\n✅ Pipeline completed in {elapsed:.2f} seconds")
