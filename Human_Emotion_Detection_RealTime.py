import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Turn off oneDNN custom operations if desired
import cv2
import tensorflow as tf
import numpy as np
# Load the model without compiling (since you are using it only for inference)
model = tf.keras.models.load_model("facialemotionmodel.h5", compile=False)

# Load the Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to preprocess the image for the model
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Open the webcam
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Labels for emotion predictions
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Create a named window to display the webcam output
cv2.namedWindow("Facial Emotion Detection")

while True:
    # Capture frame-by-frame from the webcam
    ret, im = webcam.read()
    if not ret:
        print("Error: Failed to capture image from webcam.")
        break

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Process each face found
    for (p, q, r, s) in faces:
        # Extract the face region from the grayscale image
        face_image = gray[q:q + s, p:p + r]
        cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
        
        # Resize the face region to 48x48 for the model
        face_image = cv2.resize(face_image, (48, 48))
        img = extract_features(face_image)
        
        # Predict the emotion using the pre-trained model
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]
        
        # Display the predicted emotion on the video feed
        cv2.putText(im, prediction_label, (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
    
    # Display the resulting frame in the window
    cv2.imshow("Facial Emotion Detection", im)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
webcam.release()
cv2.destroyAllWindows()
