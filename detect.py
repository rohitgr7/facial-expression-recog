# Face Recognition

# Importing the libraries
import cv2
import numpy as np
from keras.models import load_model

model = load_model('./models/updated_model.h5')
emotion_labels = {0: 'Angry', 1: 'Disguist', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
frame_input_shape = model.input_shape[1:3]

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        gray_face = gray[y:y + h, x:x + w]

        try:
            gray_face = cv2.resize(gray_face, frame_input_shape)
        except:
            continue

        gray_face = gray_face[np.newaxis, :, :, np.newaxis]
        emotion_pred = model.predict(gray_face)
        emotion_id = np.argmax(emotion_pred, axis=1)
        emotion = emotion_labels[emotion_id[0]]
        cv2.putText(frame, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

    return frame


# Doing some Face Recogntion with the webcamp
video_capture = cv2.VideoCapture(0)
window_name = 'Video'
full_screen = True

if full_screen:
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow(window_name, canvas)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
