# Course Title

Nanyang Technological University <br>
MH4510 (Statistical Learning and Data Mining)

# Project overview

This project investigates speech-based emotion recognition (SER) as a non-intrusive indicator of psychological well-being in older adults, forming the foundation for a future conversational screening assistant. Using three publicly available emotional-speech corpora (RAVDESS, CREMA-D, TESS), we benchmark classical models (Log Regression, SVM, XGBoost) and deep neural networks (CNN, BiLSTM, CNN-LSTM) using MFCC-based acoustic features.

# Significance

Singapore’s rapidly ageing population increases the need for scalable and proactive mental-health monitoring in elder-care settings. Early identification of emotional distress, particularly subsyndromal depression (SSD), is challenged by under-reporting and limited manpower. <br><br>
Our findings demonstrate technical feasibility for deployment-oriented SER and establish a modelling benchmark to support future clinical adaptation using Singapore-based elderly speech.

# Reference

CREMA_D: https://github.com/CheyneyComputerScience/CREMA-D <br>
RAVDESS: Only audio (speech) files used - data obtained from https://zenodo.org/record/1188976 <br>
TESS: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess

# .gitignore

- Audio files

# Project Main Directory Tree

```plaintext
project/
│
├── audio*speech/ # All raw audio datasets used for training/testing
│ │
│ ├── CREMA*D/ # CREMA-D emotional speech dataset (actors, single-word sentences)
│ │ ├── 1079_ITS_HAP_XX.wav # Example audio files labeled by actor + emotion
│ │ ├── 1079_ITS_FEA_XX.wav
│ │ ├── 1079_ITS_DIS_XX.wav
│ │ └── ...
│ │
│ ├── RAVDESS/ # RAVDESS dataset with emotion-labelled speech
│ │ ├── Actor_01/ # Each actor folder contains their recordings
│ │ │ ├── 03-01-01-01-01-01-01.wav
│ │ │ ├── 03-01-01-01-01-02-01.wav
│ │ │ └── ...
│ │ ├── Actor_02/
│ │ │ ├── 03-01-01-01-01-01-02.wav
│ │ │ ├── 03-01-01-01-01-02-02.wav
│ │ │ └── ...
│ │ └── ... # Actors 03–24
│ │
│ ├── TESS/ # Toronto Emotional Speech Set (female voices)
│ │ ├── OAF_angry/ # Older Adult Female (OAF) - angry emotion
│ │ │ ├── OAF_back_angry.wav
│ │ │ ├── OAF_bar_angry.wav
│ │ │ └── ...
│ │ ├── OAF_disgust/
│ │ │ ├── OAF_back_disgust.wav
│ │ │ ├── OAF_bar_disgust.wav
│ │ │ └── ...
│ │ ├── OAF\** (other emotions)
│ │ ├── YAF*angry/ # Younger Adult Female (YAF) versions
│ │ ├── YAF_disgust/
│ │ └── YAF\*\* (other emotions)
│
├── datasets_combined_no_augmented/ # Preprocessed MFCC/feature arrays (no augmentation)
│ # contains .csv and .npy files like X_train, y_train for both traditional ML and DL models
│
├── models/ # Training notebooks + saved model files
│ ├── cnn_lstm_combined_data_kfolds.ipynb # K-fold cross-validation model (no augmentation)
│ ├── cnn_lstm_aug_kfolds.ipynb # K-fold model with augmentation pipeline
│ └── ... # Other model training notebooks
│
├── crema_d_eda.ipynb # EDA and data wrangling notebook for CREMA_D dataset
├── ravdess_eda.ipynb # EDA and data wrangling notebook for RAVDESS dataset
├── tess_eda.ipynb # EDA and data wrangling notebook for TESS dataset
├── dataset.ipynb # Combined CREMA_D + RAVDESS + TESS dataset with EDA
│
└── data_preprocessing_pipeline.py # Main preprocessing script for MFCC extraction, # augmentation, normalization & dataset splitting
```

# Team members

1. Alina Xia
2. Chin Ao-Wen
3. Dhalisa
4. Elsen Ong
5. Xin Yi
