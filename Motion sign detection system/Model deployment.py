import argparse
import cv2 as cv
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Path to the trained LSTM model
MODEL_PATH = "gesture_recognition_lstm_model.hdf5"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--min_detection_confidence", type=float, default=0.8)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.8)
    parser.add_argument("--sequence_length", type=int, default=40, help="Length of sequences for LSTM")
    return parser.parse_args()

def draw_landmarks(image, landmarks, color):
    for x, y in landmarks:
        cv.circle(image, (int(x), int(y)), 5, color, -1)
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

    # Load the trained model
    model = tf.keras.models.load_model(MODEL_PATH)
    label_map = {0: "Good", 1: "Morning", 2: "ThankYou", 3: "Time"}

    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence
    )

    sequence = []
    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)  # Invert the frame horizontally
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        right_hand_landmarks = []
        left_hand_landmarks = []
        pose_landmarks = []

        frame_data = [0.0] * (21 * 2) * 2 + [0.0] * (33 * 2)  # Initialize with placeholders

        hands_present = False
        if results.right_hand_landmarks or results.left_hand_landmarks or results.pose_landmarks:
            # Process right hand landmarks (flipped to left hand)
            if results.right_hand_landmarks:
                hands_present = True
                right_hand_landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in results.right_hand_landmarks.landmark]
                norm_right_hand_landmarks = normalize_landmarks(right_hand_landmarks)
                frame_data[0:21 * 2] = [coord for lm in norm_right_hand_landmarks for coord in lm]

            # Process left hand landmarks (flipped to right hand)
            if results.left_hand_landmarks:
                hands_present = True
                left_hand_landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in results.left_hand_landmarks.landmark]
                norm_left_hand_landmarks = normalize_landmarks(left_hand_landmarks)
                frame_data[21 * 2:21 * 2 + 21 * 2] = [coord for lm in norm_left_hand_landmarks for coord in lm]

            # Process pose landmarks (shoulders, arms, etc.)
            if results.pose_landmarks:
                pose_landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in results.pose_landmarks.landmark]
                norm_pose_landmarks = normalize_landmarks(pose_landmarks)
                frame_data[21 * 2 + 21 * 2:] = [coord for lm in norm_pose_landmarks for coord in lm]

        if hands_present:
            sequence.append(frame_data)
            sequence = sequence[-args.sequence_length:]  # Keep only the last `sequence_length` frames

            if len(sequence) == args.sequence_length:
                prediction = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(prediction))

                # Display prediction
                predicted_label = label_map[np.argmax(prediction)]
                cv.putText(frame, f"Prediction: {predicted_label}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

        # Draw landmarks
        if pose_landmarks:
            frame = draw_landmarks(frame, pose_landmarks, (0, 255, 0))  # Green for pose landmarks
        if right_hand_landmarks:
            frame = draw_landmarks(frame, right_hand_landmarks, (0, 0, 255))  # Red for right hand landmarks (flipped left hand)
        if left_hand_landmarks:
            frame = draw_landmarks(frame, left_hand_landmarks, (255, 0, 0))  # Blue for left hand landmarks (flipped right hand)

        cv.imshow('Holistic Tracking', cv.cvtColor(frame, cv.COLOR_RGB2BGR))
        key = cv.waitKey(10) & 0xFF

        if key == 27:  # Press 'ESC' to quit
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
