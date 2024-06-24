import serial
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('vehicle.h5')

# Initialize serial communication
arduino = serial.Serial('COM3', 9600)

def detect_emergency_vehicle(frame):
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction[0][0] > 0.5

cap = cv2.VideoCapture(0)  # Start video capture

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if detect_emergency_vehicle(frame):
        arduino.write(b'A')  # Assuming the emergency vehicle is on road A

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()
