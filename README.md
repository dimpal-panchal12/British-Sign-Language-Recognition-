# Real-Time British Sign Language Recognition System

A real-time system that recognises 30 everyday British Sign Language (BSL) gestures using a GRU-based deep learning model. Built with TensorFlow, OpenCV, and MediaPipe, this project demonstrates the potential of AI for accessible communication.

---

## 📌 Project Overview

This project aims to improve accessibility by translating BSL into text in real time. It captures live video, extracts hand, pose, and face landmarks using MediaPipe, and feeds them into a GRU-based neural network for gesture prediction.

---

## 🛠️ Tech Stack

- Python  
- TensorFlow & Keras (deep learning)  
- OpenCV (real-time video capture)  
- MediaPipe (landmark detection)  
- NumPy, scikit-learn, Pandas  

---

## 🎯 Key Features

- Recognises 30 BSL gestures in real-time with 90% accuracy  
- Uses webcam feed and OpenCV to deliver live predictions with visual feedback  
- GRU architecture with dropout, L2 regularisation, and GridSearchCV tuning  
- Works across different lighting conditions and camera angles  
- Designed with an accessibility-first mindset  

---

## 📂 Files Included

- All source code for data collection, preprocessing, training, and live detection  
- Project report: `BSL_Project_Report.pdf`  

---

## 📥 How to Collect Data

Data was collected using OpenCV and MediaPipe by recording webcam input and extracting face, hand, and pose landmarks frame by frame. Each gesture was recorded across multiple short video sequences to ensure variation in lighting, angles, and movement.

---

## 🚀 Run Live Detection

After training, run the real-time system via:
```bash
python Real-Time\ Sign\ Language\ Detection.py
```

Make sure your webcam is enabled.

---

## 📈 Results

| Metric              | Value         |
|---------------------|---------------|
| Test Accuracy       | 90%           |
| Number of Gestures  | 30            |
| Real-Time Capable   | ✅             |

---

## 💡 Future Work

- Extend to full BSL vocabulary  
- Add multi-camera support  
- Deploy as desktop or browser-based app  
- Integrate with text-to-speech tools

---

## 🤝 Acknowledgements

Built as part of a postgraduate dissertation project focused on improving AI accessibility for the deaf and speech-impaired communities.
