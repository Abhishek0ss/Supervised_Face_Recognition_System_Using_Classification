# Supervised_Face_Recognition_System_Using_Classification
🎭 Supervised Face Recognition System Using HOG + Machine Learning

This project implements a complete face recognition system using classical machine learning models and HOG (Histogram of Oriented Gradients) feature extraction.
It supports face detection, feature engineering, multi-model evaluation, unknown-face detection, and image-based predictions.

🚀 1. Project Overview

The goal of this project is to classify human faces using supervised ML algorithms.
The system includes:

Face detection using dlib

HOG feature extraction

Outlier removal

Train/Validation/Test split

Model comparison (LogReg, KNN, SVM)

ROC-based thresholding for unknown face detection

Prediction system with bounding boxes

The final model (SVM with RBF kernel) delivers strong accuracy and generalization.

🎯 2. Key Features
✔ Face Detection

dlib’s HOG-based detector

Consistent padding applied for stable cropping

Works on grayscale and color images

✔ HOG Feature Extraction

All images normalized using CLAHE

Resize: 128 × 128

HOG parameters:

orientations=9

pixels_per_cell=(32, 32)

cells_per_block=(2, 2)

Produces a compact feature vector for ML models

✔ Outlier Filtering

Removes bottom 2% of samples using low L2 norm

Helps eliminate corrupted and blank images

✔ Unknown Face Detection

Uses ROC curve Youden’s J statistic to compute optimal per-class thresholds

If highest probability < threshold → output = Unknown

✔ Visual Analysis

Confusion matrix

ROC curves for each model

Class distribution plot

Similarity heatmap of HOG features

🧠 3. Machine Learning Models Implemented
Logistic Regression

Baseline softmax classifier
Applied with L2 normalization and StandardScaler

KNN (k = 3)

Instance-based learner
Good for small datasets

SVM (RBF kernel) — Final Recommended Model

Pipeline components:

StandardScaler → L2 Normalizer → PCA(0.98 variance) → SVM(RBF)


Best performing model

Handles nonlinear face patterns well

Balanced class weights improve fairness across classes

📊 4. Model Performance

Your code automatically displays:

Classification report

Accuracy, macro recall, macro precision, macro F1

Confusion matrix

ROC curves per class

Probability distribution for unknown detection

(Exact numbers depend on your dataset.)

🧱 5. Dataset Format

Each folder represents a class/identity.

Supported extensions: .jpg, .jpeg, .png, .pgm

🔧 6. How the Pipeline Works
Step-by-step flow:

Load dataset

Detect face + crop + apply padding

Extract HOG features

Remove outlier samples

Split into Train / Validation / Test sets

Train LR, KNN, SVM models

Compute ROC curve thresholds

Predict and classify uploaded test images

Mark low-confidence predictions as Unknown

Save trained models using joblib

🧪 7. Running Predictions

Use the function:

predict_image()


Upload any face image

System detects the face

HOG feature extracted

SVM probability analyzed

If probability < threshold → Unknown

Output displayed on the image using OpenCV

Example output:

Prediction: person1 (0.82)


or for low confidence:

Prediction: Unknown (prob=0.21)

💾 8. Saving Artifacts

The script automatically saves:

pipeline.joblib          → Trained SVM model
hog_svm.joblib           → ROC thresholds for Unknown detection
Hog_labelMap.npy         → Mapping of class IDs to names

⚙️ 9. Requirements

Install dependencies:

pip install numpy pandas scikit-learn dlib opencv-python seaborn matplotlib joblib


🚀 10. Future Enhancements

Replace HOG features with deep neural embeddings (ArcFace / FaceNet)

Add MTCNN for better facial alignment

Build a Streamlit / Flask web UI

Improve unknown detection with distance-based thresholds (Cosine/L2)

Add data augmentation (flip, blur, noise)

📝 11. Conclusion

This project demonstrates a fully operational supervised face recognition system using classical ML and HOG features.
The pipeline is clean, modular, and easy to extend for real-world applications.
