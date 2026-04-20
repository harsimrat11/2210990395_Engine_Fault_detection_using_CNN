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
This section explains the complete process for running the project from start to finish.
---
## Recommended Environment
The project is recommended to run on **Google Colab** because the script uses `/content/` paths for uploaded datasets and audio files. It can also run on a local system (Windows/Linux/macOS) after changing file paths.
### Why Google Colab?
- Free GPU/CPU access  
- Easy file upload  
- No installation issues on local machine  
- Faster training performance  
- Best suited for TensorFlow projects  
---
## System Requirements
Minimum recommended:
- Python 3.9 or above  
- 8 GB RAM  
- Intel i5 / Ryzen 5 or better  
- 5 GB free storage  
- Internet connection (for installing packages)
For faster training:
- GPU enabled Google Colab runtime
---
## Step 1: Install Dependencies
Run the following command:
```bash
pip install librosa tensorflow scikit-learn pandas matplotlib seaborn scikit-image soundfile

Required Libraries

* TensorFlow – deep learning framework
* Librosa – audio processing
* NumPy – numerical operations
* Pandas – data handling
* Matplotlib / Seaborn – graphs
* Scikit-learn – metrics and validation
* Scikit-image – image resizing
* SoundFile – audio file reading

⸻

Step 2: Prepare Required Files

Place or upload the following files into your project folder
(or /content/ if using Google Colab):

Mandatory Files

* engine_fault_cnn.py
* ai_mechanic_dataset.zip

Optional Additional Audio Files

Extra .mp3 samples may be added for custom testing, such as:

* Normal engine idle sound
* Misfire sound
* Knocking sound
* Ticking sound
* Belt squeal sound
* Spark plug issue sound

⸻

Step 3: Folder Structure

Recommended structure:

Project Folder/
│── engine_fault_cnn.py
│── ai_mechanic_dataset.zip
│── sample1.mp3
│── sample2.mp3
│── results/
│── README.md

⸻

Step 4: Run the Project

Execute the main script:

python engine_fault_cnn.py

If using Google Colab:

1. Upload all required files
2. Open notebook / code file
3. Run all cells sequentially

⸻

Step 5: What Happens Automatically

After execution starts, the script performs the following pipeline:

Dataset Processing

* Detects and extracts ai_mechanic_dataset.zip
* Loads labelled engine audio files
* Loads manually added .mp3 samples
* Combines all audio data into one dataset

Audio Preprocessing

* Converts audio to 22050 Hz
* Converts stereo to mono
* Splits recordings into 3-second clips
* Applies zero-padding where required

Data Augmentation

To increase dataset diversity:

* Time shifting
* Gaussian noise addition
* Pitch shifting

Feature Extraction

* Generates Mel-spectrograms
* Converts to decibel scale
* Resizes to 128 × 128
* Normalizes image values

Model Training

Trains:

5-Class CNN Model

* Normal Engine
* Combustion Fault
* Mechanical Fault
* Belt / Accessory Fault
* Background Noise

Binary CNN Model

* Normal
* Faulty

Evaluation

Automatically generates:

* Accuracy graph
* Loss graph
* Confusion matrix
* ROC curve
* Classification report
* 5-Fold Cross Validation score

⸻

Step 6: Output Files Generated

Saved Models

* engine_fault_multiclass.keras
* engine_fault_binary.keras

Result Images / Reports

Stored in /results folder:

* accuracy.png
* loss.png
* confusion_matrix.png
* ROC_curve.png
* mel_spectrogram.png
* 5Fold_cross_validation.jpg
* normal_vs_faulty.jpg
* melspectogram_comparison.jpg
* results_tables.pdf

⸻

Default Training Configuration

Parameter	Value
Sample Rate	22050 Hz
Segment Length	3 Seconds
Input Shape	128 × 128 × 1
Batch Size	32
Epochs	12
Optimizer	Adam
Validation	5-Fold Cross Validation
Train/Test Split	80% / 20%

⸻

Approximate Runtime

Device	Time
Google Colab GPU	15-35 min
Google Colab CPU	30–60 min
Local CPU	Depends on hardware

⸻

Troubleshooting

Dataset not found

Make sure ai_mechanic_dataset.zip is in the same folder.

MP3 file errors

Check filenames and paths inside the script.

TensorFlow not installed

pip install tensorflow

Memory issues

Use Google Colab GPU runtime.

⸻

Important Notes

* If running locally, replace /content/... file paths with your system path.
* Results may vary slightly because of training randomness.
* GPU gives faster training than CPU.

⸻

Final Command Summary

pip install librosa tensorflow scikit-learn pandas matplotlib seaborn scikit-image soundfile
python engine_fault_cnn.py

## Author

**Harsimrat Singh**  
Chitkara University
