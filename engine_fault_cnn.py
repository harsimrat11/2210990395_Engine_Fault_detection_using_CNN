# =============================================================================
# ENGINE FAULT CLASSIFICATION USING CNN
# Audio-Based Multi-Class and Binary Fault Detection
# =============================================================================

# =============================================================================
# SECTION 1: ENVIRONMENT SETUP & IMPORTS
# =============================================================================

import os
import zipfile
import json
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import glob
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import librosa.display

# Set deterministic random seeds for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
print(f"Random seeds set to {SEED} for reproducibility.")

# Audio processing constants
SAMPLE_RATE = 22050       # Hz
TARGET_DURATION = 3       # seconds


# =============================================================================
# SECTION 2: DATASET LOADING
# Unzip dataset, load labels, and merge with manually uploaded MP3s
# =============================================================================

zip_file_path = 'ai_mechanic_dataset.zip'
extraction_dir = 'ai_mechanic_dataset'

# Unzip dataset if not already extracted
if os.path.exists(zip_file_path):
    if not os.path.exists(extraction_dir):
        os.makedirs(extraction_dir)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extraction_dir)
        print(f"Dataset '{zip_file_path}' unzipped to '{extraction_dir}'.")
    else:
        print(f"Extraction directory '{extraction_dir}' already exists. Skipping unzipping.")
else:
    print(f"Error: Zipped dataset file '{zip_file_path}' not found.")

# Locate info.labels file within the extracted directory
label_file_paths = glob.glob(os.path.join(extraction_dir, '**', 'info.labels'), recursive=True)
if not label_file_paths:
    raise FileNotFoundError(f"info.labels not found in {extraction_dir} or its subdirectories.")
label_file_path = label_file_paths[0]

with open(label_file_path, 'r') as f:
    labels_data_json = json.load(f)
print(f"Labels loaded from '{label_file_path}'.")

# Manually uploaded MP3 files with their labels
manual_mp3_files = [
    "/content/2012 Mustang GT 5.0 Coyote engine rattle or knock.mp3",
    "/content/2004 Mustang GT SOUND OF SPARK PLUG BACKING OUT. (motor is from 2000, not a romeo 2v).mp3",
    "/content/Stock 2020 Mustang Gt PP1 Cold Start #cartok #mustang #s550.mp3",
    "/content/download.mp3",
    "/content/ASMR_ Ford Mustang GT500 Idling _ Speed Therapy _ Ford [vvMfryHBStU].mp3",
    "/content/squeaking noise coming from the alternator belt __ (1).mp3",
    "/content/What a misfire sounds like..mp3",
    "/content/Squealwhistle noise mustang.mp3",
    "/content/squeaking noise coming from the alternator belt __.mp3",
    "/content/5.3 V8 ticking noise, lifters or spark plugs_.mp3",
    "/content/This is what it sounds like if you have a blown spark plug on a Ford F150..mp3"
]

manual_labels_mapping = {
    "2012 Mustang GT 5.0 Coyote engine rattle or knock.mp3":                                        "rattle_knock",
    "2004 Mustang GT SOUND OF SPARK PLUG BACKING OUT. (motor is from 2000, not a romeo 2v).mp3":    "spark_plug_issue",
    "Stock 2020 Mustang Gt PP1 Cold Start #cartok #mustang #s550.mp3":                              "normal",
    "download.mp3":                                                                                  "unknown",
    "ASMR_ Ford Mustang GT500 Idling _ Speed Therapy _ Ford [vvMfryHBStU].mp3":                     "normal",
    "squeaking noise coming from the alternator belt __ (1).mp3":                                   "squeaking_belt",
    "What a misfire sounds like..mp3":                                                               "misfire",
    "Squealwhistle noise mustang.mp3":                                                               "squeal_whistle",
    "squeaking noise coming from the alternator belt __.mp3":                                       "squeaking_belt",
    "5.3 V8 ticking noise, lifters or spark plugs_.mp3":                                            "ticking_lifters",
    "This is what it sounds like if you have a blown spark plug on a Ford F150..mp3":               "spark_plug_issue"
}

# Build unified audio data list
all_audio_data = []
dataset_root_dir = os.path.dirname(label_file_path)

# Load entries from zipped dataset
if 'files' in labels_data_json:
    for file_entry in labels_data_json['files']:
        relative_filepath = file_entry['path']
        label = file_entry['label']['label']
        full_path = os.path.join(dataset_root_dir, relative_filepath)

        if os.path.exists(full_path):
            all_audio_data.append({'filepath': full_path, 'label': label})
        else:
            # Fallback: search by filename
            potential_paths = glob.glob(os.path.join(extraction_dir, '**', os.path.basename(relative_filepath)), recursive=True)
            if potential_paths:
                all_audio_data.append({'filepath': potential_paths[0], 'label': label})
            else:
                print(f"Warning: Could not find {relative_filepath}")
else:
    print("Error: 'files' key not found in labels JSON.")

# Load manually uploaded MP3 entries
for filepath in manual_mp3_files:
    filename = os.path.basename(filepath)
    label = manual_labels_mapping.get(filename, 'unknown')
    if os.path.exists(filepath):
        all_audio_data.append({'filepath': filepath, 'label': label})
    else:
        print(f"Warning: Manually uploaded file not found: {filepath}")

df_audio = pd.DataFrame(all_audio_data)
print(f"\nTotal audio samples loaded: {len(df_audio)}")
print("\nClass Distribution:")
print(df_audio['label'].value_counts())


# =============================================================================
# SECTION 3: AUDIO AUGMENTATION FUNCTIONS
# Time shifting, Gaussian noise addition, Pitch shifting
# =============================================================================

def time_shift(audio, sr, shift_range=0.2):
    """Shifts the audio by a random percentage of its length."""
    roll_amount = int(sr * random.uniform(-shift_range, shift_range) * len(audio) / sr)
    return np.roll(audio, roll_amount)

def add_gaussian_noise(audio, noise_factor=0.005):
    """Adds Gaussian noise to the audio signal."""
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def pitch_shift(audio, sr, n_steps=2):
    """Shifts the pitch of the audio by a random number of semitones."""
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=random.randint(-n_steps, n_steps))

print("Augmentation functions defined: time_shift, add_gaussian_noise, pitch_shift.")


# =============================================================================
# SECTION 4: AUDIO PREPROCESSING & SEGMENTATION
# Segment audio into 3-second clips, pad short clips, apply augmentations
# =============================================================================

processed_audio_data = []
segment_length_samples = SAMPLE_RATE * TARGET_DURATION

for index, row in df_audio.iterrows():
    filepath = row['filepath']
    label = row['label']

    try:
        audio, sr = librosa.load(filepath, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        continue

    total_segments = int(np.ceil(len(audio) / segment_length_samples))

    for i in range(total_segments):
        start = i * segment_length_samples
        end = min((i + 1) * segment_length_samples, len(audio))
        segment = audio[start:end]

        # Zero-pad if shorter than target duration
        if len(segment) < segment_length_samples:
            segment = np.pad(segment, (0, segment_length_samples - len(segment)), 'constant')

        # Store original segment
        processed_audio_data.append({'audio_segment': segment, 'label': label})

        # Apply augmentations (skip 'unknown' labels)
        if label != 'unknown':
            processed_audio_data.append({'audio_segment': time_shift(segment, sr),     'label': label})
            processed_audio_data.append({'audio_segment': add_gaussian_noise(segment), 'label': label})
            processed_audio_data.append({'audio_segment': pitch_shift(segment, sr),    'label': label})

df_processed_audio = pd.DataFrame(processed_audio_data)
print(f"\nTotal processed audio segments (original + augmented): {len(df_processed_audio)}")
print("\nClass Distribution of processed segments:")
print(df_processed_audio['label'].value_counts())


# =============================================================================
# SECTION 5: MEL-SPECTROGRAM FEATURE EXTRACTION
# Compute 128-band Mel-spectrograms, convert to dB, resize to 128x128, normalize
# =============================================================================

mel_spectrograms = []
labels = []

for index, row in df_processed_audio.iterrows():
    audio_segment = row['audio_segment']
    label = row['label']

    # Compute Mel-spectrogram
    S = librosa.feature.melspectrogram(y=audio_segment, sr=SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=128)

    # Convert to decibels
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Resize to 128x128 pixels
    resized_S_dB = resize(S_dB, (128, 128), anti_aliasing=True, preserve_range=True, order=1).astype(np.float32)

    # Per-sample min-max normalization
    min_val, max_val = np.min(resized_S_dB), np.max(resized_S_dB)
    if max_val == min_val:
        normalized_S_dB = np.zeros_like(resized_S_dB)
    else:
        normalized_S_dB = (resized_S_dB - min_val) / (max_val - min_val)

    mel_spectrograms.append(normalized_S_dB)
    labels.append(label)

# Reshape for CNN input: (samples, height, width, channels)
X = np.array(mel_spectrograms).reshape(-1, 128, 128, 1)
y = np.array(labels)

print(f"Mel-spectrogram array shape (X): {X.shape}")
print(f"Labels array shape (y): {y.shape}")


# =============================================================================
# SECTION 6: DATASET PREPARATION FOR CLASSIFICATION
# Filter unknowns, encode labels for multi-class and binary tasks
# =============================================================================

# Filter out 'unknown' labels
known_indices = np.where(y != 'unknown')[0]
X_filtered = X[known_indices]
y_filtered = y[known_indices]
print(f"After filtering unknowns — X: {X_filtered.shape}, y: {y_filtered.shape}")

# --- Multi-class label encoding ---
label_encoder_multi = LabelEncoder()
y_multi = label_encoder_multi.fit_transform(y_filtered)
multi_class_labels_mapping = dict(zip(
    label_encoder_multi.classes_,
    label_encoder_multi.transform(label_encoder_multi.classes_)
))
print(f"\nMulti-class label mapping: {multi_class_labels_mapping}")

# --- Binary label encoding (0 = Normal/Background/Idling, 1 = Faulty) ---
y_binary = np.array([
    0 if label.lower() in ('normal', 'background noise', 'idling') else 1
    for label in y_filtered
])

print(f"\nMulti-class label distribution:")
unique_m, counts_m = np.unique(y_multi, return_counts=True)
for lbl, cnt in zip(unique_m, counts_m):
    name = label_encoder_multi.inverse_transform([lbl])[0]
    print(f"  {name}: {cnt}")

print(f"\nBinary label distribution:")
unique_b, counts_b = np.unique(y_binary, return_counts=True)
for lbl, cnt in zip(unique_b, counts_b):
    name = 'Normal/Background/Idling' if lbl == 0 else 'Faulty'
    print(f"  {name}: {cnt}")


# =============================================================================
# SECTION 7: CNN MODEL ARCHITECTURE
# Shared builder function for both multi-class and binary models
# =============================================================================

def build_cnn_model(input_shape, num_classes):
    """Builds and compiles a CNN model for audio classification."""
    model = Sequential([
        InputLayer(input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# =============================================================================
# SECTION 8: GROUPED (REDUCED) MULTI-CLASS MODEL
# Remap fine-grained labels into 5 meaningful fault groups
# =============================================================================

# Group mapping: 0=Normal, 1=Combustion Fault, 2=Mechanical Fault, 3=Belt Noise, 4=Background
group_mapping = {
    'normal':                       0,
    'normal engine inside cabin':   0,
    'idling':                       0,
    'misfire':                      1,
    'spark_plug_issue':             1,
    'oil cap off engine inside cabin': 1,
    'rattle_knock':                 2,
    'ticking_lifters':              2,
    'air leak':                     2,
    'air leak engine inside cabin': 2,
    'squeaking_belt':               3,
    'squeal_whistle':               3,
    'Background Noise':             4,
    'background noise':             4,
    'testing':                      4
}

# Stratified 80-20 train-test split for multi-class
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
for train_idx, test_idx in sss.split(X_filtered, y_multi):
    X_train_multi, X_test_multi = X_filtered[train_idx], X_filtered[test_idx]
    y_train_multi, y_test_multi = y_multi[train_idx], y_multi[test_idx]

print(f"Train set: {X_train_multi.shape}, Test set: {X_test_multi.shape}")

# Remap to grouped labels
old_labels_train = label_encoder_multi.inverse_transform(y_train_multi)
old_labels_test  = label_encoder_multi.inverse_transform(y_test_multi)
y_train_reduced  = np.array([group_mapping[l] for l in old_labels_train])
y_test_reduced   = np.array([group_mapping[l] for l in old_labels_test])

print("\nGrouped class distribution (train):")
unique_g, counts_g = np.unique(y_train_reduced, return_counts=True)
print(dict(zip(unique_g, counts_g)))

# Train grouped 5-class model
input_shape = X_filtered.shape[1:]
num_classes_reduced = 5
reduced_model = build_cnn_model(input_shape, num_classes_reduced)

history = reduced_model.fit(
    X_train_multi,
    y_train_reduced,
    epochs=12,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)


# =============================================================================
# SECTION 9: MODEL EVALUATION — GROUPED MULTI-CLASS
# Test accuracy, classification report, confusion matrix
# =============================================================================

test_loss, test_acc = reduced_model.evaluate(X_test_multi, y_test_reduced)
print(f"\nTest Accuracy (Grouped 5-Class): {test_acc:.4f}")

y_pred_reduced = np.argmax(reduced_model.predict(X_test_multi), axis=1)
print("\nClassification Report (Grouped 5-Class):")
print(classification_report(y_test_reduced, y_pred_reduced))

# Confusion Matrix
cm = confusion_matrix(y_test_reduced, y_pred_reduced)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix — Grouped 5-Class")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()


# =============================================================================
# SECTION 10: TRAINING CURVES
# Training vs Validation Accuracy and Loss over epochs
# =============================================================================

# Accuracy curve
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'],     label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Loss curve
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'],     label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# =============================================================================
# SECTION 11: ROC-AUC CURVES — MULTI-CLASS
# One-vs-Rest ROC curve for each of the 5 grouped classes
# =============================================================================

y_test_bin = label_binarize(y_test_reduced, classes=[0, 1, 2, 3, 4])
y_score    = reduced_model.predict(X_test_multi)

plt.figure(figsize=(8, 6))
for i in range(5):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i} AUC = {roc_auc:.2f}")

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-Class ROC Curve (One-vs-Rest)")
plt.legend()
plt.tight_layout()
plt.show()


# =============================================================================
# SECTION 12: 5-FOLD CROSS-VALIDATION — MULTI-CLASS
# Evaluate model stability using stratified k-fold CV on training set
# =============================================================================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
accuracy_multi_cv = []

print("Starting 5-fold stratified cross-validation for multi-class model...")

for fold, (train_idx_fold, val_idx_fold) in enumerate(skf.split(X_train_multi, y_train_reduced)):
    print(f"\n--- Fold {fold + 1}/5 ---")

    X_train_fold = X_train_multi[train_idx_fold]
    X_val_fold   = X_train_multi[val_idx_fold]
    y_train_fold = y_train_reduced[train_idx_fold]
    y_val_fold   = y_train_reduced[val_idx_fold]

    fold_model = build_cnn_model(input_shape, num_classes_reduced)
    fold_model.fit(X_train_fold, y_train_fold, epochs=8, batch_size=32,
                   validation_data=(X_val_fold, y_val_fold), verbose=1)

    _, acc = fold_model.evaluate(X_val_fold, y_val_fold, verbose=0)
    accuracy_multi_cv.append(acc)
    print(f"Fold {fold + 1} Validation Accuracy: {acc:.4f}")

mean_cv_accuracy = np.mean(accuracy_multi_cv)
print(f"\nMean 5-Fold Cross-Validation Accuracy (Multi-class): {mean_cv_accuracy:.4f}")


# =============================================================================
# SECTION 13: SAMPLE MEL-SPECTROGRAM VISUALISATION
# =============================================================================

plt.figure(figsize=(6, 4))
librosa.display.specshow(X_train_multi[0].squeeze(), cmap='magma')
plt.title("Example Mel-Spectrogram (Training Sample)")
plt.colorbar()
plt.tight_layout()
plt.show()


# =============================================================================
# SECTION 14: SAVE MODELS
# =============================================================================

reduced_model.save("engine_fault_multiclass.keras")
print("Multi-class model saved as 'engine_fault_multiclass.keras'.")

# Binary model — train on X_filtered with y_binary
sss_bin = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
for train_idx_bin, test_idx_bin in sss_bin.split(X_filtered, y_binary):
    X_train_bin, X_test_bin = X_filtered[train_idx_bin], X_filtered[test_idx_bin]
    y_train_bin, y_test_bin_labels = y_binary[train_idx_bin], y_binary[test_idx_bin]

binary_model = build_cnn_model(input_shape, num_classes=2)
binary_model.fit(X_train_bin, y_train_bin, epochs=12, batch_size=32,
                 validation_split=0.1, verbose=1)

bin_loss, bin_acc = binary_model.evaluate(X_test_bin, y_test_bin_labels)
print(f"\nBinary Model Test Accuracy: {bin_acc:.4f}")

binary_model.save("engine_fault_binary.keras")
print("Binary model saved as 'engine_fault_binary.keras'.")
