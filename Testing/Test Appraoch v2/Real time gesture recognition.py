import cv2 as cv
import numpy as np
import tensorflow as tf
import mediapipe as mp
import itertools
import copy
from collections import deque

# Load the trained model
model = tf.keras.models.load_model('keypoint_classifier.hdf5')

# Define the class labels
labels = ['Time', 'Hello', 'Shaking']

# Preprocess the landmarks and point history
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize the webcam
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

# Function to calculate landmarks
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    return [[landmark.x * image_width, landmark.y * image_height]
            for landmark in landmarks.landmark]

# Function to draw landmarks on image
def draw_landmarks(image, landmark_list):
    for index, (x, y) in enumerate(landmark_list):
        cv.circle(image, (int(x), int(y)), 5, (255, 255, 255), -1)
        cv.circle(image, (int(x), int(y)), 5, (0, 0, 0), 1)
    return image

# Function to draw point history
def draw_point_history(image, point_history):
    for index, points in enumerate(point_history):
        for (x, y) in points:
            if x != 0 and y != 0:
                cv.circle(image, (int(x), int(y)), 1 + int(index / 2), (152, 251, 152), 2)
    return image

# Real-time gesture prediction
point_history = deque(maxlen=16)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv.flip(frame, 1)
    debug_image = copy.deepcopy(frame)
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    if results.multi_hand_landmarks:
        all_hand_landmarks = []
        fingertip_points = []
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)
            all_hand_landmarks.extend(landmark_list)
            fingertip_points.extend([landmark_list[i] for i in [4, 8, 12, 16, 20]])

            debug_image = draw_landmarks(debug_image, landmark_list)

        point_history.append(fingertip_points)
        debug_image = draw_point_history(debug_image, point_history)

        if len(all_hand_landmarks) == 42:
            normalized_landmarks = pre_process_landmark(all_hand_landmarks)
            normalized_point_history = pre_process_landmark(fingertip_points)

            X_landmarks = np.array([normalized_landmarks])
            X_fingertips = np.array([normalized_point_history])

            prediction = model.predict([X_landmarks, X_fingertips])
            predicted_class_index = np.argmax(prediction)
            predicted_class_label = labels[predicted_class_index]

            cv.putText(debug_image, f'Prediction: {predicted_class_label}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv.LINE_AA)

    cv.imshow('Hand Gesture Recognition', debug_image)
    if cv.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()
