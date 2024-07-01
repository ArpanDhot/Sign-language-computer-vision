import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Initialize Mediapipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Load the trained model
model = tf.keras.models.load_model('sign_language_model.keras')

# Load label encoder classes
label_encoder_classes = np.load('classes.npy')


# Function to preprocess hand landmarks
def preprocess_landmarks(landmarks):
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    landmarks = (landmarks - np.min(landmarks, axis=0)) / (np.max(landmarks, axis=0) - np.min(landmarks, axis=0))
    return landmarks.flatten()


# Initialize webcam
cap = cv2.VideoCapture(0)

sequence = []
max_sequence_length = 150

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Preprocess landmarks
            processed_landmarks = preprocess_landmarks(hand_landmarks)
            sequence.append(processed_landmarks)

            # Ensure the sequence length is consistent
            if len(sequence) > max_sequence_length:
                sequence.pop(0)

            # Make predictions if we have enough frames
            if len(sequence) == max_sequence_length:
                input_data = np.expand_dims(sequence, axis=0)
                predictions = model.predict(input_data)
                predicted_label = label_encoder_classes[np.argmax(predictions)]

                # Display the prediction on the frame
                cv2.putText(frame, f'Sign: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Sign Language Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
