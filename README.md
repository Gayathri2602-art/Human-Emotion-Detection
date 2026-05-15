# Human-Emotion-Detection

😊 Facial Emotion Detection
A real-time facial emotion recognition system using Deep Learning and Computer Vision that detects and classifies human emotions from live webcam feed and static images.

🎯 Features

    🎥 Real-time emotion detection via webcam

     🖼️ Static image prediction support

     😤 Detects 7 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise

     ⚡ Fast and lightweight using a pre-trained .h5 model
 
     📊 Visualize predictions using Matplotlib


🛠️ Tech Stack

       Python
       TensorFlow / Keras — Deep learning model

      OpenCV — Face detection & webcam feed

     NumPy — Image preprocessing

      Matplotlib — Image visualization

     Haar Cascade Classifier — Face detection

🧠 How It Works

Webcam captures live video frames

Haar Cascade detects faces in each frame

Detected face is cropped and resized to 48×48 grayscale

Pre-trained CNN model predicts the emotion

Emotion label is displayed on the video feed in real-time

Dataset- https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset

Python Version 3.8.10

Code Editor- Visual Studio Code
