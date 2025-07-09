# Urdu Deepfake Audio Detection

A detection framework for Urdu audio deepfakes, evaluating both classical machine learning and modern deep learning models using the ACL 2024 Urdu Deepfake Audio Dataset.

---

## Project Overview

This project uses a publicly released dataset containing both bonafide and synthetic Urdu audio. The dataset is designed for phonemic coverage and includes spoofing attacks generated via Tacotron and VITS TTS. Models implemented include:

- Logistic Regression  
- Support Vector Machine (SVM)  
- Simple Perceptron  
- Multilayer Perceptron (MLP)  
- LSTM / BiLSTM  
- Convolutional Neural Networks (VGG16, ResNet18)

---

## Repository Structure

├── data/
│ ├── Bonafide_Part1/
│ ├── Bonafide_Part2/
│ ├── Tacotron/
│ └── VITS_TTS/
├── notebooks/
│ ├── AASIST_DFDAU.ipynb
│ ├── CNN_VGG16_ResNet18_DFDAU.ipynb
│ ├── LSTM_BiLSTM_DFDAU.ipynb
│ ├── MLP_DFDAU.ipynb
│ ├── Perceptron_DFDAU.ipynb
│ ├── SVM_DFDAU.ipynb
│ └── logistic_regression_DFDAU.ipynb
├── src/
├── requirements.txt
└── README.md


---

## Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/urdu-deepfake-audio-detection.git
   cd urdu-deepfake-audio-detection
Install dependencies:


pip install -r requirements.txt
Download the dataset and place it in the data/ folder with the existing structure.

Models & Results
Model	Feature	Accuracy	Notes
Logistic Regression	Mel Spectrogram	92.7%	Balanced errors
SVM	Mel Spectrogram	97.0%	Best among classical ML
Simple Perceptron	Mel Spectrogram	90.7%	Weaker recall on fake samples
MLP (2 hidden layers)	MFCC	97.9%	Slight edge over Mel features
LSTM	—	50.9%	Unstable and biased prediction
BiLSTM	—	99.2%	High precision and recall
ResNet18 (CNN)	Mel Spectrogram	99.94%	Best overall performance
VGG16 (CNN)	Mel Spectrogram	97.3%	Mild overfitting observed

Running Experiments
Example: Train SVM on Mel features:

python src/train_svm.py \
  --data_dir data/ \
  --feature mel_spectrogram \
  --output_dir outputs/svm_mel/
You can also run and evaluate other models via the Jupyter notebooks in the notebooks/ directory.

Feature Extraction
Supported feature types:

MFCC

Mel Spectrogram

Chroma

Feature extraction is implemented within the notebooks or src/ scripts, and saved as .npy files.

Evaluation
All models are evaluated using:

Accuracy, Precision, Recall, F1-Score, AUC

Confusion matrices and ROC curves

Detailed analysis and visualizations are provided within the notebooks.

Future Work
Feature fusion (e.g., combining MFCC + Mel)

Transformer-based and attention models

Speaker-independent data splits and cross-dataset validation

Audio data augmentation (noise, pitch, speed variations)

License
This project is available under the MIT License. See LICENSE for details.

Contact
Abdulhaadii
Email: abdulhaadii641@gmail.com

Contributions, issue reports, and pull requests are welcome!



