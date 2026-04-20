# Acoustic Signal-Based Non-Intrusive Engine Fault Detection Using Mel-Spectrograms and Convolutional Neural Networks


A non-intrusive engine fault detection system that identifies engine conditions from audio recordings — no diagnostic tools or special hardware required, just sound.

Engine audio is converted into Mel-spectrograms and fed into a Convolutional Neural Network (CNN) trained to distinguish between normal operation and various fault types.

---

## Overview

Internal combustion engines produce consistent sound patterns during normal operation. When faults develop — misfires, knocking, worn belts, spark plug issues — these patterns change in measurable ways. Instead of relying on OBD tools or physical inspection, this project analyses those acoustic changes directly.

The pipeline: record engine audio → segment into 3-second clips → augment → extract Mel-spectrogram features → classify with CNN.

---

## Fault Categories

| Class | Examples |
|-------|----------|
| Normal Operation | Smooth running, idling |
| Combustion Fault | Misfire, spark plug issue |
| Mechanical Fault | Knocking, ticking, rattle |
| Belt / Accessory Fault | Squealing, belt noise |
| Background Noise | Non-engine ambient sounds |

---

## Dataset

Two sources were combined to build the training dataset:

- **AI Mechanic Dataset** — a public dataset of labelled engine audio samples covering multiple fault conditions
- **Real-world recordings** — 11 additional engine audio clips collected from publicly available sources and manually labelled based on audio content

**Augmentation techniques applied:**
- Time shifting — small forward/backward shift of the audio signal
- Gaussian noise injection — simulates real-world recording conditions
- Pitch variation — slight pitch adjustment to increase diversity

Augmentation was applied only to labelled samples to expand the effective training set size.

---

## Model Architecture

Each audio clip is converted to a normalised 128×128 Mel-spectrogram image, which is used as CNN input.

| Layer | Details |
|-------|---------|
| Input | 128 × 128 × 1 |
| Conv2D | 32 filters, 3×3, ReLU |
| MaxPooling | 2×2 |
| Conv2D | 64 filters, 3×3, ReLU |
| MaxPooling | 2×2 |
| Flatten | — |
| Dense | 128 units, ReLU |
| Dropout | 0.5 |
| Output | 5 classes, Softmax |

**Training config:** Adam optimizer · Sparse categorical crossentropy · Batch size 32 · 12 epochs · 80-20 stratified train-test split · 5-fold cross-validation

---

## Results

**Mean 5-Fold Cross-Validation Accuracy: ~86–87%**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Normal Engine | 1.00 | 0.92 | 0.96 |
| Combustion Fault | 0.92 | 0.92 | 0.92 |
| Mechanical Fault | 0.69 | 0.98 | 0.81 |
| Belt / Accessory Fault | 1.00 | 0.91 | 0.95 |
| Background Noise | 0.90 | 0.62 | 0.73 |

**ROC-AUC (One-vs-Rest):** Class 0 = 0.99 · Class 1 = 1.00 · Class 2 = 0.97 · Class 3 = 1.00 · Class 4 = 0.97

**Comparison with baseline:**

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| SVM + MFCC (baseline) | 78% | 0.76 |
| CNN + Mel-Spectrogram (this work) | 86% | 0.84 |

---

## Output Visualisations

All result plots are saved in the `/results` folder.

| File | Description |
|------|-------------|
| `accuracy.png` | Training vs Validation Accuracy across epochs |
| `loss.png` | Training vs Validation Loss across epochs |
| `confusion_matrix.png` | Confusion matrix for the 5-class model |
| `ROC_curve.png` | Multi-class ROC curves (one-vs-rest) |
| `mel_spectrogram.png` | Example Mel-spectrogram from training data |
| `results_tables.pdf` | Full classification report |
| `training_vs_validation_loss.jpg` |Training vs Validation Loss for actual 25 epochs |
| `training_vs_validation_accuracy.jpg` |Training vs Validation Accuracy for actual 25 epochs |

---

## How to Run

**Install dependencies**
```bash
pip install librosa tensorflow scikit-learn pandas matplotlib seaborn scikit-image soundfile
```

**Prepare files**

Place the following in your project folder (or `/content/` if using Google Colab):
- `ai_mechanic_dataset.zip`
- All engine audio `.mp3` files

**Run**
```bash
python engine_fault_cnn.py
```

Trained models will be saved as:
- `engine_fault_multiclass.keras`
- `engine_fault_binary.keras`

---

## Sample Audio

A few representative engine audio clips are included in the `/Sample Audio` folder for reference.

---

## Author

**Harsimrat Singh**  
Chitkara University
