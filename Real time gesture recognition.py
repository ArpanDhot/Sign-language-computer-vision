import argparse
import cv2 as cv
import mediapipe as mp
import numpy as np
import tensorflow as tf

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    return parser.parse_args()

def draw_landmarks(image, landmark_list):
    for x, y in landmark_list:
        cv.circle(image, (int(x), int(y)), 5, (255, 255, 255), -1)
        cv.circle(image, (int(x), int(y)), 5, (0, 0, 0), 1)
    return image

def normalize_landmarks(landmarks):
    max_value = max(list(map(abs, [coord for landmark in landmarks for coord in landmark])))
    if max_value == 0:
        return landmarks  # Avoid division by zero
    normalized_landmarks = [(x / max_value, y / max_value) for x, y in landmarks]
    return normalized_landmarks

def main():
    args = get_args()

    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence
    )

    # Load the trained model
    model = tf.keras.models.load_model('keypoint_classifier.hdf5')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Define class names
    class_names = ['A', 'B', 'C', 'D']

    confidence_threshold = 0.95  # Set confidence threshold to 80%

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)  # Invert the frame horizontally
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        landmark_list = []
        if results.multi_hand_landmarks:
            frame_data = [0.0] * 84  # Initialize with placeholders
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label
                hand_status = 0.1 if hand_label == 'Left' else 0.2

                landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in hand_landmarks.landmark]
                landmark_list.extend(landmarks)

                # Normalize landmarks
                norm_landmarks = normalize_landmarks(landmarks)

                # Flatten and insert into the correct position in frame_data
                if hand_label == 'Left':
                    frame_data[:42] = [coord for lm in norm_landmarks for coord in lm]
                else:
                    frame_data[42:] = [coord for lm in norm_landmarks for coord in lm]

            # Prepare data for prediction
            landmarks_array = np.array(frame_data).reshape(1, -1)

            # Make prediction
            prediction = model.predict(landmarks_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence_score = np.max(prediction)

            # Display the predicted class on the frame if confidence is above threshold
            if confidence_score > confidence_threshold:
                class_name = class_names[predicted_class]
                cv.putText(frame, f'Class: {class_name} ({confidence_score:.2f})', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)

        # Draw landmarks
        if landmark_list:
            frame = draw_landmarks(frame, landmark_list)

        cv.imshow('Hand Tracking', cv.cvtColor(frame, cv.COLOR_RGB2BGR))
        key = cv.waitKey(10) & 0xFF

        if key == 27:  # Press 'ESC' to quit
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
