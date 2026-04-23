# 😷 Face Mask Detection System

<div align="center">

![Face Mask Detection](https://img.shields.io/badge/AI-Face%20Mask%20Detection-blue?style=for-the-badge&logo=tensorflow)
![Accuracy](https://img.shields.io/badge/Accuracy-97%25-brightgreen?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?style=for-the-badge&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-green?style=for-the-badge&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**An AI-powered real-time face mask detection system built with Deep Learning.**  
Detects mask/no-mask in real-time via webcam or uploaded images with bounding boxes and confidence scores.

[🔗 Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) • [👤 GitHub](https://github.com/Soni875612) • [💼 LinkedIn](https://www.linkedin.com/in/soni-devi-131a9938b/)
[![LeetCode](https://img.shields.io/badge/LeetCode-soni__2007-orange?style=flat-square&logo=leetcode)](https://leetcode.com/u/soni_2007/)

</div>

---

## 📌 About The Project

This project was developed as part of an **NSTI College Project** to build a practical AI safety surveillance system. The system uses a Convolutional Neural Network (CNN) trained on annotated face images to classify whether a person is wearing a face mask or not — in real time.

With increasing importance of public health safety, this system can be deployed at entry points, workplaces, or public spaces to monitor mask compliance automatically.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎥 **Real-Time Detection** | Live webcam feed with instant classification |
| 📸 **Image Upload** | Upload any image and get instant results |
| 📊 **Analytics Dashboard** | Track compliance stats per session |
| 🎯 **97% Accuracy** | High precision model trained on 853 images |
| 🟢🔴 **Visual Feedback** | Green box = with mask, Red box = without mask |
| 📈 **Confidence Score** | Shows prediction confidence % on screen |
| 🌐 **Web Interface** | Clean, modern browser-based UI via Flask |
| ⚡ **Fast Inference** | ~40-50ms per frame on CPU |

---

## 🎬 Demo
## Emages
<img width="800" height="639" alt="Screenshot 2026-04-23 184733" src="https://github.com/user-attachments/assets/bca1be61-2646-499b-800d-1fea80487b03" />

<img width="811" height="650" alt="Screenshot 2026-04-23 091530" src="https://github.com/user-attachments/assets/4df46b0f-8145-495d-883d-0763354db39f" />

<img width="814" height="640" alt="Screenshot 2026-04-23 163705" src="https://github.com/user-attachments/assets/d154b5fc-792f-4dac-9042-4e43592de4a5" />
<img width="821" height="652" alt="Screenshot 2026-04-23 163647" src="https://github.com/user-attachments/assets/66c67183-c107-4424-8cd2-a9d99f11823e" />


## Vedio


```
With Mask ✅          Without Mask ❌
┌─────────────┐       ┌─────────────┐
│  with_mask  │       │without_mask │
│    100.0%    │       │   99.4%   │
└─────────────┘       └─────────────┘
 GREEN bounding box    RED bounding box
```

---

## 🧠 Model Performance

| Metric | with_mask | without_mask | Overall |
|--------|-----------|--------------|---------|
| Precision | 0.99 | 0.91 | — |
| Recall | 0.98 | 0.95 | — |
| F1-Score | 0.98 | 0.93 | — |
| **Accuracy** | — | — | **97%** |

> Trained for **9 epochs** on **853 images** using TensorFlow/Keras CNN architecture.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.13 |
| **ML Framework** | TensorFlow 2.21, Keras |
| **Computer Vision** | OpenCV 4.13 |
| **Web Framework** | Flask 3.0 |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Data Format** | PASCAL VOC (XML Annotations) |
| **Model Format** | Keras `.h5` |

---

## 📁 Project Structure

```
face_mask_detection/
│
├── 📂 Dataset/
│   ├── images/          # 853 training images
│   └── annotations/     # XML annotation files (PASCAL VOC)
│
├── 📂 saved_model/
│   └── face_mask_model.h5   # Trained Keras model
│
├── 📂 static/
│   ├── css/             # Stylesheets
│   └── uploads/         # Uploaded images (runtime)
│
├── 📂 templates/
│   ├── index.html       # Main UI
│   └── result.html      # Detection result page
│
├── 🐍 app.py               # Flask web server (main entry point)
├── 🐍 train_model.py        # Model training script
├── 🐍 detect_live.py        # Live webcam detection
├── 🐍 detect_image.py       # Image-based detection
├── 🐍 evaluate_model.py     # Model evaluation & metrics
├── 🐍 analytics.py          # Analytics & session tracking
├── 🐍 config.py             # Configuration & constants
├── 🐍 utils.py              # Helper functions
└── 📄 requirements.txt      # Python dependencies
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+
- Webcam (for live detection)
- pip

### Step 1 — Clone the Repository
```bash
git clone https://github.com/Soni875612/face-mask-detection.git
cd face-mask-detection
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the Web App
```bash
python app.py
```

### Step 4 — Run the Web App
```bash
python detect_live.py
```

---

## 🎥 Live Detection (Standalone)

To run webcam detection without the web interface:
```bash
python detect_live.py
```
> Press **Q** to quit the detection window.

---

## 🏋️ Train the Model (Optional)

To retrain on your own dataset:
```bash
python train_model.py
```
> Trained model will be saved to `saved_model/face_mask_model.h5`

---

## 📊 Evaluate Model
```bash
python evaluate_model.py
```

---

## 🗃️ Dataset

- **Source:** [Face Mask Detection Dataset — Kaggle](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
- **Format:** PASCAL VOC (XML annotations)
- **Classes:** `with_mask`, `without_mask`
- **Total Images:** 853
- **Train/Val Split:** 80% / 20%

---

## 🚀 How It Works

```
📷 Camera Frame
      ↓
🔍 Haar Cascade Face Detection
      ↓
✂️  ROI Extraction (Face Crop)
      ↓
🔄 Preprocessing (224×224, Normalize)
      ↓
🧠 CNN Model Prediction
      ↓
🎨 Bounding Box + Label + Confidence
      ↓
🖥️  Display / Web Response
```

---

## ⚠️ Known Limitations

- Dark lighting reduces detection accuracy
- Black/dark colored masks may show lower confidence (model trained on white/blue surgical masks)
- Live browser stream requires `localhost` due to HTTP camera restrictions
- Camera index may need to be changed from `0` to `1` depending on system

---

## 📋 Requirements

```
tensorflow>=2.12.0
opencv-python>=4.8.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
flask>=3.0.0
flask-cors>=4.0.0
Pillow>=10.0.0
seaborn>=0.12.0
```

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---


---

## 👤 Author

<div align="center">

**Soni**  
*NSTI College Project*

[![GitHub](https://img.shields.io/badge/GitHub-Soni875612-black?style=flat-square&logo=github)](https://github.com/Soni875612)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Soni%20Devi-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/soni-devi-131a9938b/)
[![LeetCode](https://img.shields.io/badge/LeetCode-soni__2007-orange?style=flat-square&logo=leetcode)](https://leetcode.com/u/soni_2007/)

</div>

---

<div align="center">

⭐ **If you found this project helpful, please give it a star!** ⭐

</div>
