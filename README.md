# 🎭 Emotion Detection using Speech (RAVDESS Dataset)

##  Project Overview
This project focuses on **Emotion Detection** from speech signals using the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset**.  
The goal is to classify different human emotions from audio features using both **Machine Learning (Random Forest)** and **Deep Learning (CNN)** models.

-  Developed in **Google Colab Notebook**  
-  Deployed using **Streamlit**  
-  Used both **ML and DL approaches**  
-  Dataset: **RAVDESS Speech Emotional Dataset**  

---

##  Tech Stack & Libraries
- **Machine Learning**: Random Forest Classifier  
- **Deep Learning**: Convolutional Neural Network (CNN)  
- **Frameworks & Libraries**:
  - `TensorFlow / Keras`
  - `Librosa` (for audio feature extraction like MFCCs, Mel spectrograms)
  - `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`
  - `Scikit-learn`
  - `Streamlit` (for deployment)

---

##  Model Performance
- **Random Forest Classifier (ML Model)** → 🎯 Accuracy: **68%**  
- **Convolutional Neural Network (DL Model)** → 🎯 Accuracy: **80%**

👉 Clearly, the CNN model performs significantly better for this task due to its ability to capture spatial patterns in extracted audio features.

---

##  Project Workflow
1. **Dataset Preparation**  
   - Used **RAVDESS emotional speech dataset**  
   - Extracted features (MFCC, chroma, Mel spectrogram) using `librosa`

2. **Model Training**  
   - Trained Random Forest for baseline ML approach  
   - Designed a CNN model in TensorFlow/Keras for Deep Learning approach  

3. **Evaluation**  
   - Compared accuracy, loss curves, and confusion matrices  
   - CNN achieved better accuracy and generalization  

4. **Deployment**  
   - Saved trained CNN model (`cnn_model.h5`)  
   - Built an interactive **Streamlit web app** for real-time emotion detection  

---

## 📂 Folder Structure
- Emotion_detection_RAVDESS/
-  │── app.py # Streamlit app
-  │── cnn_model.h5 # Trained CNN model
-  │── requirements.txt # Dependencies
-  │── Emotion_detection.ipynb # Colab notebook (training & experiments)
-  │── README.md # Project Documentation

## ▶️ Running the Project Locally
1. Clone the repo:
   ```bash
   git clone https://github.com/EC-Arpita/Emotion_detection_RAVDESS.git
   cd Emotion_detection_RAVDESS
2. Install Dependencies:
```
pip install -r requirements.txt
   
```
3. Run Streamlit app:
```
Streamlit run app.py

```