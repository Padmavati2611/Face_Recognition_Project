# face_recognition.py👤

## 📌 Project Description

This project is a **Face Recognition System** that detects and identifies human faces using computer vision and machine learning 
techniques. It can be used for applications like attendance systems, security, and identity verification.

## 🚀 Features

* Face detection in images and video
* Face recognition using trained models
* Real-time recognition using webcam
* High accuracy with trained dataset
* Easy-to-use interface

## 🛠️ Technologies Used

* Python
* OpenCV
* NumPy
* (Optional) TensorFlow / Keras / dlib

## 📂 Project Structure

```
face-recognition/
│── dataset/          # Images for training
│── trainer/          # Trained model files
│── recognizer.py     # Face recognition script
│── dataset_creator.py# Capture face images
│── trainer.py        # Train the model
│── README.md
```

## ⚙️ Installation

1. Clone the repository:

```
git clone https://github.com/padmavati2611/face-recognition.git
```

2. Install dependencies:

```
pip install opencv-python numpy
```

3. (Optional) Install additional libraries:

```
pip install dlib face-recognition
```

## ▶️ Usage

1. Collect dataset:

```
python dataset_creator.py
```

2. Train the model:

```
python trainer.py
```

3. Run face recognition:

```
python recognizer.py
```

## 📸 Output

* Detects faces in real-time
* Displays name of recognized person
* ![IMG-20260315-WA0029](https://github.com/user-attachments/assets/a316f993-e6a8-4014-bd06-683126b5fe73)




