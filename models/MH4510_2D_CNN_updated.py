#!/usr/bin/env python
# coding: utf-8

# 
# 2D-CNN Baseline Training Summary (Combined RAVDESS + CREMA-D + TESS Dataset)
# 
# Data:
# - Total samples after merging three acted speech datasets:
#     Train = 6,390
#     Val   = 2,130
#     Test  = 2,130
# - Each input is an MFCC feature map of shape (T, 30), where T varies by sample.
# - Sequences were zero-padded to the max length in the val set → final input shape = (455, 30, 1).
# - Seven emotion classes; noticeable imbalance → class weights applied.
# 
# Model:
# - Time-centric 2D-CNN with 4 convolutional blocks + Global Average Pooling + 128-unit Dense layer.
# - Total parameters ≈ 222K (lightweight; suitable baseline architecture).
# - Dropout + BatchNorm used to stabilize training and reduce overfitting.
# 
# The 2D-CNN model achieved a test accuracy of 53.99% and a balanced accuracy of 50.93% across seven emotional classes. While the model performs substantially better than random chance (≈14%), the performance distribution across classes reveals uneven generalization.
# 
# The confusion matrix indicates that certain classes, particularly class 4, are recognized with relatively high precision and recall (F1 = 0.72), suggesting the CNN successfully learned distinguishing acoustic cues for that emotion. Conversely, classes 2, 6, and 7 display weaker recall (≤0.31), implying that the model frequently misclassifies these emotions or underrepresents them during training. This discrepancy may stem from class imbalance, feature overlap in the MFCC space, or insufficient temporal modeling capability inherent in CNN-only architectures.
# 
# From the classification report, precision and recall values vary widely:
# 
# Class 1 demonstrates high recall (0.80) but low precision (0.44), suggesting over-prediction.
# 
# Class 6 shows high precision (0.77) but low recall (0.26), implying the model predicts this emotion rarely but accurately when it does.
# 
# Class 7 has very low recall (0.19), likely due to its smaller representation in the dataset.
# 
# These results confirm that the inclusion of Δ (delta) features enhanced the model’s sensitivity to short-term spectral variations, leading to a significant improvement over the previous MFCC-only baseline (22% → 54% accuracy). However, the CNN remains limited in capturing long-term temporal dependencies that characterize emotional progression in speech.
# 
# To address this, future model iterations should consider hybrid architectures such as CNN–BiLSTM or CNN–BiGRU, which combine convolutional feature extraction with recurrent sequence modeling. Additionally, implementing focal loss or class-weighted sampling could mitigate imbalance-related bias and improve recall for minority emotion classes. Adjusting the dropout rate to 0.3–0.35 and employing a cyclical learning rate schedule (e.g., 1e-4–1e-3) may also enhance training stability and convergence.
# 
# Overall, this configuration establishes a solid baseline for the augmented dataset, demonstrating the CNN’s ability to extract meaningful affective features from MFCC and delta representations, while highlighting the need for temporal sequence modeling to achieve state-of-the-art performance.
# 

# In[1]:


# ======================= SER 2D-CNN (MFCC / Δ / Δ²) — CLEAN BASELINE =======================
# Works with variable-length .npy (dtype=object) and fixed-length arrays.
# Uses channel-aware inputs: (time, 30, C) with C in {1,2,3}.
# Padding = 95th percentile of train+val lengths (reduces zero-padding bias).
# Regularization: Dropout=0.5, L2=1e-4, label smoothing, ReduceLROnPlateau.
# Evaluation: accuracy, balanced accuracy, confusion matrix, classification report.

# --- 0) Optional installs (uncomment if needed) ---
# %pip install -q numpy scikit-learn tensorflow==2.16.1

import os, sys, math, json, time, random
import numpy as np
from collections import Counter
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score

# ------------------------
# 1) CONFIG
# ------------------------
SEED = 234
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Choose which dataset folder to use and where it lives
DATASET_CHOICE = "combined"   # "combined" -> datasets_combined_ravdess_crema_tess ; "augmented" -> datasets_combined_augmented

if DATASET_CHOICE == "combined":
    DATA_DIR = r"D:\MH4510\project\datasets_combined_ravdess_crema_tess"
elif DATASET_CHOICE == "augmented":
    DATA_DIR = r"D:\MH4510\project\datasets_combined_augmented"
else:
    raise ValueError("DATASET_CHOICE must be 'combined' or 'augmented'.")

# Channel selection (feature ablation)
#   'A' = MFCC only  (recommended to start)
#   'B' = MFCC + Δ
#   'C' = MFCC + Δ + Δ²
CHANNEL_MODE = 'B'  # change to 'B' or 'C' to include delta / delta-delta

# Training
EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3
DROPOUT = 0.35
L2_WD = 1e-4
LABEL_SMOOTH = 0.05
PERCENTILE_PAD = 95  # pad length chosen from train+val lengths

# Filenames (assumes standard naming)
X_train_path = os.path.join(DATA_DIR, "X_deep_train.npy")
y_train_path = os.path.join(DATA_DIR, "y_deep_train.npy")
X_val_path   = os.path.join(DATA_DIR, "X_deep_val.npy")
y_val_path   = os.path.join(DATA_DIR, "y_deep_val.npy")
X_test_path  = os.path.join(DATA_DIR, "X_deep_test.npy")
y_test_path  = os.path.join(DATA_DIR, "y_deep_test.npy")

for p in [X_train_path, y_train_path, X_val_path, y_val_path, X_test_path, y_test_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing file: {p}")

# ------------------------
# 2) LOADING HELPERS
# ------------------------
def load_npy(path: str):
    """
    Loads .npy that may be a regular ndarray or an object array (list of arrays).
    Returns either:
      - ndarray of shape (N, T, F) for fixed-length
      - list of (T_i, F) arrays for variable-length
    """
    arr = np.load(path, allow_pickle=True)
    # Object array case (variable-length sequences)
    if arr.dtype == object:
        return [np.asarray(x) for x in arr]
    # Already numeric tensor
    return np.asarray(arr)

def get_lengths(X):
    """Return list of sequence lengths T for either list-of-arrays or ndarray (N, T, F)."""
    if isinstance(X, list):
        return [x.shape[0] for x in X]
    else:
        return [X.shape[1]] * X.shape[0]

def ensure_list(X):
    """Ensure X is list of (T_i, F)."""
    if isinstance(X, list):
        return X
    # ndarray (N, T, F)
    return [X[i] for i in range(X.shape[0])]

def pad_to_length(X_list, L):
    """
    Zero-pad/truncate list of (T_i, F) to (N, L, F).
    """
    N = len(X_list)
    F = X_list[0].shape[1]
    out = np.zeros((N, L, F), dtype=np.float32)
    for i, x in enumerate(X_list):
        t = min(L, x.shape[0])
        out[i, :t, :] = x[:t, :]
    return out

# ------------------------
# 3) LOAD SPLITS
# ------------------------
X_tr = load_npy(X_train_path)
y_tr = np.load(y_train_path, allow_pickle=True)
X_va = load_npy(X_val_path)
y_va = np.load(y_val_path, allow_pickle=True)
X_te = load_npy(X_test_path)
y_te = np.load(y_test_path, allow_pickle=True)

# Convert labels to a consistent type (str safe)
y_tr = np.array([str(x) for x in y_tr])
y_va = np.array([str(x) for x in y_va])
y_te = np.array([str(x) for x in y_te])

N_tr = len(X_tr) if isinstance(X_tr, list) else X_tr.shape[0]
N_va = len(X_va) if isinstance(X_va, list) else X_va.shape[0]
N_te = len(X_te) if isinstance(X_te, list) else X_te.shape[0]

# Infer feature dimension F
def infer_F(X):
    if isinstance(X, list):
        return X[0].shape[1]
    else:
        return X.shape[2]
F_infer = infer_F(ensure_list(X_tr))

print("=== Raw shapes (before padding) ===")
print(f"Train: N={N_tr}, sample[0] shape={ensure_list(X_tr)[0].shape}, F={F_infer}")
print(f"Val  : N={N_va}, sample[0] shape={ensure_list(X_va)[0].shape}")
print(f"Test : N={N_te}, sample[0] shape={ensure_list(X_te)[0].shape}")

# ------------------------
# 4) CHOOSE PAD LENGTH = PERCENTILE(train+val)
# ------------------------
lens_tr = get_lengths(X_tr)
lens_va = get_lengths(X_va)
L_target = int(np.percentile(lens_tr + lens_va, PERCENTILE_PAD))
X_tr_pad = pad_to_length(ensure_list(X_tr), L_target)
X_va_pad = pad_to_length(ensure_list(X_va), L_target)
X_te_pad = pad_to_length(ensure_list(X_te), L_target)

# ------------------------
# 5) TRAIN-ONLY Z-SCORE NORMALIZATION (per feature column)
# ------------------------
def fit_z(X):  # X: (N, L, F)
    NL, F = X.shape[0]*X.shape[1], X.shape[2]
    XF = X.reshape(NL, F)
    mu = XF.mean(axis=0, keepdims=True)
    sd = XF.std(axis=0, keepdims=True) + 1e-8
    return mu, sd

def apply_z(X, mu, sd):
    NL, F = X.shape[0]*X.shape[1], X.shape[2]
    XF = X.reshape(NL, F)
    YF = (XF - mu)/sd
    return YF.reshape(X.shape)

mu, sd = fit_z(X_tr_pad)
X_tr_z = apply_z(X_tr_pad, mu, sd).astype(np.float32)
X_va_z = apply_z(X_va_pad, mu, sd).astype(np.float32)
X_te_z = apply_z(X_te_pad, mu, sd).astype(np.float32)

# ------------------------
# 6) CHANNEL RESHAPE to (time, 30, C) from (time, F) where F may be >= 90
# ------------------------
def split_channels_from_F(X, mode='A'):
    """
    X: (N, L, F). Assumes first 90 columns = [MFCC0..29 | Δ0..29 | Δ²0..29] if available.
    For augmented sets with extra features (ZCR, RMSE, etc.), we simply ignore >90 columns for the CNN.
    mode: 'A' MFCC only, 'B' MFCC+Δ, 'C' MFCC+Δ+Δ²
    Returns: (N, L, 30, C)
    """
    N, L, F = X.shape
    if mode == 'A':
        if F < 30:
            raise ValueError("Need at least 30 features for MFCC-only mode.")
        m = X[:, :, 0:30]
        return m[..., None]  # (N,L,30,1)

    if mode == 'B':
        if F < 60:
            raise ValueError("Need at least 60 features for MFCC+Δ mode.")
        m = X[:, :, 0:30]
        d = X[:, :, 30:60]
        return np.concatenate([m[..., None], d[..., None]], axis=-1)  # (N,L,30,2)

    if mode == 'C':
        if F < 90:
            raise ValueError("Need at least 90 features for MFCC+Δ+Δ² mode.")
        m = X[:, :, 0:30]
        d = X[:, :, 30:60]
        a = X[:, :, 60:90]
        return np.concatenate([m[..., None], d[..., None], a[..., None]], axis=-1)  # (N,L,30,3)

    raise ValueError("CHANNEL_MODE must be 'A', 'B', or 'C'.")

X_tr_4d = split_channels_from_F(X_tr_z, CHANNEL_MODE)
X_va_4d = split_channels_from_F(X_va_z, CHANNEL_MODE)
X_te_4d = split_channels_from_F(X_te_z, CHANNEL_MODE)

print("=== Model input tensor ===")
print("Train:", X_tr_4d.shape, "Val:", X_va_4d.shape, "Test:", X_te_4d.shape)
C = X_tr_4d.shape[-1]

# ------------------------
# 7) LABELS → indices → one-hot + class weights
# ------------------------
labels_sorted = sorted(list(set(y_tr)))  # strings
lab2idx = {lab:i for i, lab in enumerate(labels_sorted)}
idx2lab = {i:lab for lab,i in lab2idx.items()}

y_tr_i = np.array([lab2idx[s] for s in y_tr], dtype=np.int32)
y_va_i = np.array([lab2idx[s] for s in y_va], dtype=np.int32)
y_te_i = np.array([lab2idx.get(s, -1) for s in y_te], dtype=np.int32)
if np.any(y_te_i < 0):
    raise ValueError("Found test labels not seen in training labels.")

K = len(labels_sorted)
y_tr_oh = tf.keras.utils.to_categorical(y_tr_i, K)
y_va_oh = tf.keras.utils.to_categorical(y_va_i, K)

# Class weights (balanced)
cw_arr = compute_class_weight(class_weight="balanced", classes=np.arange(K), y=y_tr_i)
class_weights = {i: float(cw_arr[i]) for i in range(K)}
print("Class weights:", class_weights)

# ------------------------
# 8) MODEL: Time-centric 2D-CNN
#     - Kernels: (5×3) (taller in time)
#     - Pooling: (2×1) (along time only)
#     - Dropout + L2 + label smoothing
# ------------------------
def make_timecentric_cnn(input_shape, num_classes, p=0.5, wd=1e-4):
    inp = layers.Input(shape=input_shape)  # (L, 30, C)

    x = layers.Conv2D(32, (5,3), padding='same', kernel_regularizer=regularizers.l2(wd))(inp)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2,1))(x); x = layers.Dropout(p)(x)

    x = layers.Conv2D(64, (5,3), padding='same', kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2,1))(x); x = layers.Dropout(p)(x)

    x = layers.Conv2D(128, (5,3), padding='same', kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=(2,1))(x); x = layers.Dropout(p)(x)

    # narrow conv to mix channels/coefficients
    x = layers.Conv2D(128, (3,1), padding='same', kernel_regularizer=regularizers.l2(wd))(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(wd), name='penultimate')(x)
    x = layers.Dropout(p)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inp, out)

input_shape = X_tr_4d.shape[1:]  # (L, 30, C)
model = make_timecentric_cnn(input_shape, K, p=DROPOUT, wd=L2_WD)
loss_obj = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
              loss=loss_obj, metrics=['accuracy'])
model.summary()

# ------------------------
# 9) CALLBACKS & TRAIN
# ------------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint("best_2dcnn_timecentric.keras", monitor='val_accuracy', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=1)
]

history = model.fit(
    X_tr_4d, y_tr_oh,
    validation_data=(X_va_4d, y_va_oh),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks, verbose=2
)

# ------------------------
# 10) EVALUATION (TEST)
# ------------------------
probs = model.predict(X_te_4d, verbose=0)
pred_idx = probs.argmax(axis=1)

acc = accuracy_score(y_te_i, pred_idx)
bal_acc = balanced_accuracy_score(y_te_i, pred_idx)
print(f"\n*** 2D-CNN Test Accuracy: {acc:.4f} ***")
print(f"Balanced Accuracy: {bal_acc:.4f}")

cm = confusion_matrix(y_te_i, pred_idx)
print("\nConfusion matrix:\n", cm)

print("\nClassification report:")
print(classification_report(y_te_i, pred_idx, digits=4, zero_division=0, target_names=[idx2lab[i] for i in range(K)]))

# ------------------------
# 11) (OPTIONAL) CNN → XGBoost ENSEMBLE (uncomment if xgboost installed)
# ------------------------
# try:
#     from xgboost import XGBClassifier
#     feat_model = tf.keras.Model(model.input, model.get_layer('penultimate').output)
#     Z_tr = feat_model.predict(X_tr_4d, verbose=0)
#     Z_va = feat_model.predict(X_va_4d, verbose=0)
#     Z_te = feat_model.predict(X_te_4d, verbose=0)
#     xgb = XGBClassifier(
#         n_estimators=600, max_depth=5, learning_rate=0.05,
#         subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
#         random_state=SEED, n_jobs=-1
#     )
#     xgb.fit(np.vstack([Z_tr, Z_va]), np.hstack([y_tr_i, y_va_i]))
#     pred_ens = xgb.predict(Z_te)
#     acc_ens = accuracy_score(y_te_i, pred_ens)
#     bal_acc_ens = balanced_accuracy_score(y_te_i, pred_ens)
#     print(f"\n*** CNN+XGBoost Ensemble Test Accuracy: {acc_ens:.4f} ***")
#     print(f"Balanced Accuracy (Ensemble): {bal_acc_ens:.4f}")
# except Exception as e:
#     print("Ensemble step skipped:", e)
# ======================= END =======================
##

