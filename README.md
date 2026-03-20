# 🎓 MH4510 — Statistical Learning and Data Mining

### Nanyang Technological University (NTU)

# 🔍 Project overview

This project explores **Speech-based Emotion Recognition (SER)** as a non-intrusive indicator of **psychological well-being in older adults**. By leveraging 3 widely used emotional speech corpora — **RAVDESS**, **CREMA-D**, **TESS** — we benchmark:

- 📊 **Classical ML Models:** Logistic Regression, SVM, XGBoost
- 🤖 **Deep Learning Models:** CNN, BiLSTM, CNN-LSTM
- 🎵 **Acoustic Features:** MFCCs + delta/delta-delta + ZCR + RMS Energy

This serves as the technical foundation for the possible future development of a **conversational screening assistant** in elder-care settings.

# ✨ Significance

Singapore’s fast-ageing population increases the need for **scalable mental-health monitoring tools**.  
Early detection of emotional distress (e.g., **Subsyndromal Depression, SSD**) is often hindered by:

- ⛔ Under-reporting by seniors
- 👩‍⚕️ Limited healthcare manpower
- 🗣️ Subtle emotional expression patterns

Our project demonstrates the technical feasibility of SER for deployment-oriented screening, and provides a benchmark for **future adaptation to Singapore-based elderly speech data**.

📄 **Full Report:**  
👉 [Click here to view the full project report (PDF)](./MH4510_Report_Team_Winners.pdf)

# 📚 Reference Datasets

- **CREMA-D:** https://github.com/CheyneyComputerScience/CREMA-D
- **RAVDESS:** (speech only) https://zenodo.org/record/1188976
- **TESS:** https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess

# 🛑 .gitignore

- Audio files
- venv
- npy files

# 📁 Project Structure

```markdown
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

# 👥 Team Members

1. Alina Xia
2. Chin Ao-Wen
3. Dhaliisa Valen
4. Elsen Ong
5. Xin Yi
