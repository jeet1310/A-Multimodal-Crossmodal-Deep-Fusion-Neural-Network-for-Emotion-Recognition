# 🎭 Unified Multimodal–Crossmodal Emotion Recognition Framework  

This repository contains the implementation of my research project **“A Unified Multimodal–Crossmodal Deep Fusion Framework for Emotion Recognition”**, developed as part of my undergraduate research at **NMIMS, Navi Mumbai**.

📁 **Code and project files:** [Google Drive Folder](https://drive.google.com/drive/folders/1tr1vZar04EafuIgGa7M9-xjFTe9t2R0B?usp=sharing)

---

## 🧠 Project Description  

This project focuses on building an advanced **emotion recognition system** that integrates **audio, visual, and textual modalities** to capture the full spectrum of human emotions.  

Traditional unimodal models — using only speech, facial expression, or text — often fail to recognize emotions accurately under varying real-world conditions. To overcome these limitations, this framework fuses information from all three modalities using a **deep learning–based gated and attention-driven fusion mechanism**.

The model extracts:  
- 🎤 **Speech features** using *Librosa* (MFCC, Chroma, Pitch, Spectral Descriptors)  
- 😀 **Facial emotion features** using *DeepFace*  
- 💬 **Text sentiment features** using *Vosk ASR* for transcription and *BERT* for semantic understanding  

All three feature sets are combined through a **fusion network** that jointly learns emotional context, resulting in highly accurate and robust predictions.

---

## 🎯 Highlights  

- **Trimodal integration:** Combines speech, facial, and text features for emotion recognition.  
- **Deep learning fusion:** Uses gated and attention-based mechanisms for improved accuracy.  
- **High performance:** Achieved **96.88% accuracy** on the *RAVDESS* dataset.  
- **Statistical validation:** Model predictions were statistically verified for reliability.  
- **Applications:** Human-computer interaction, mental health monitoring, educational technology, and social computing.  

---

## ⚙️ Tech Stack  

| Category | Tools / Frameworks |
|-----------|-------------------|
| Programming | Python 3.8 |
| Deep Learning | TensorFlow, Keras |
| Audio Processing | Librosa, MoviePy |
| Facial Emotion Analysis | DeepFace, OpenCV |
| Text Analysis | Vosk ASR, BERT (Transformers) |
| Data Visualization | Matplotlib, Seaborn |
| Dataset | RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) |

---

## 🧩 Setup Instructions  

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/unified-multimodal-emotion-recognition.git
   cd unified-multimodal-emotion-recognition
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the RAVDESS dataset or your own emotion dataset and place it in the `data/` folder.

4. (Optional) Download the full project code and pre-trained models from Google Drive:  
   🔗 [Project Files](https://drive.google.com/drive/folders/1tr1vZar04EafuIgGa7M9-xjFTe9t2R0B?usp=sharing)

5. Run training:
   ```bash
   python main.py --mode train
   ```

6. Evaluate the model:
   ```bash
   python main.py --mode evaluate
   ```

---

## 📊 Results  

| Metric | Value |
|--------|--------|
| Accuracy | **96.88%** |
| Precision (Macro Avg.) | 97.24% |
| Recall (Macro Avg.) | 96.67% |
| F1-Score (Macro Avg.) | 96.93% |

The model shows consistent performance across all seven basic emotions — *anger, disgust, fear, happiness, sadness, surprise,* and *neutral.*

---

## 💡 Applications  

- 🧠 **Mental health monitoring:** Emotion tracking for depression or anxiety detection  
- 🧑‍💻 **Human-computer interaction:** Emotion-aware interfaces and chatbots  
- 🏫 **Education:** Adaptive learning systems that respond to student emotions  
- 📱 **Social media analytics:** Sentiment and emotion detection in videos and posts  
- 🩺 **Healthcare:** Assistive emotion recognition for therapy and rehabilitation  

---

## 📬 Contact  

**Author:** Jeet Jain  
**Email:** [jainjeet1310@gmail.com](mailto:jainjeet1310@gmail.com)  
**Institution:** STME, NMIMS, Navi Mumbai  

If you find this project helpful or wish to collaborate, feel free to reach out or fork the repository.  

---

**📎 Project Files:** [Google Drive Link](https://drive.google.com/drive/folders/1tr1vZar04EafuIgGa7M9-xjFTe9t2R0B?usp=sharing)
