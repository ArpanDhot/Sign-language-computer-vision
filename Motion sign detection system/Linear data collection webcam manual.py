import argparse
import csv
import cv2 as cv
import mediapipe as mp


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
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

    with open('hand_arm_shoulder_landmarks.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["Label", "Series"] + [f"Left_LM_{i}_x" for i in range(21)] + [f"Left_LM_{i}_y" for i in range(21)] + \
                 [f"Right_LM_{i}_x" for i in range(21)] + [f"Right_LM_{i}_y" for i in range(21)] + \
                 [f"Pose_{i}_x" for i in range(33)] + [f"Pose_{i}_y" for i in range(33)]
        writer.writerow(header)

        recording = False
        label = ""
        session_data = []
        sequence_length = args.sequence_length
        series_number = 1

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

            if recording:
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
                    session_data.append([label, series_number] + frame_data)

                # Ensure we have at least `sequence_length` frames
                if len(session_data) == sequence_length:
                    for seq_frame_data in session_data:
                        writer.writerow(seq_frame_data)
                    series_number += 1
                    session_data = []  # Clear the session data after saving

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
            elif key == ord('a'):  # Press 'A' to start recording
                recording = True
                session_data = []
                label = input("Enter the label for this recording: ")
                series_number = 1
                print("Recording started...")
            elif key == ord('s'):  # Press 'S' to stop recording
                recording = False
                print("Recording stopped.")
            elif key == ord('d'):  # Press 'D' to save recording
                recording = False
                if len(session_data) == sequence_length:
                    for seq_frame_data in session_data:
                        writer.writerow(seq_frame_data)
                    print(f"Data saved with label: {label}")
                else:
                    print("No complete sequence to save.")
                session_data = []

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
