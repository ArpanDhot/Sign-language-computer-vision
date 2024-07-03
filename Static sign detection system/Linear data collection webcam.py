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

    with open('../hand_landmarks.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["Label", "Hand_Status"] + [f"Left_LM_{i}_x" for i in range(21)] + [f"Left_LM_{i}_y" for i in
                                                                                     range(21)] + [f"Right_LM_{i}_x" for
                                                                                                   i in range(21)] + [
                     f"Right_LM_{i}_y" for i in range(21)]
        writer.writerow(header)

        recording = False
        label = ""
        session_data = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv.flip(frame, 1)  # Invert the frame horizontally
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            landmark_list = []
            if recording:
                frame_data = [label, 0.0] + [0.0] * (21 * 2) * 2  # Initialize with placeholders

                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        hand_label = handedness.classification[0].label
                        hand_status = 0.1 if hand_label == 'Left' else 0.2

                        landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in hand_landmarks.landmark]
                        landmark_list.extend(landmarks)

                        # Normalize landmarks
                        norm_landmarks = normalize_landmarks(landmarks)

                        if hand_label == 'Left':
                            frame_data[1] = hand_status if frame_data[1] == 0.0 else 0.3
                            frame_data[2:2 + 21 * 2] = [coord for lm in norm_landmarks for coord in lm]
                        else:
                            frame_data[1] = hand_status if frame_data[1] == 0.0 else 0.3
                            frame_data[2 + 21 * 2:] = [coord for lm in norm_landmarks for coord in lm]

                # Check if the entire row is not all zeros
                if any(frame_data[2:]):
                    session_data.append(frame_data)

            # Draw landmarks
            if landmark_list:
                frame = draw_landmarks(frame, landmark_list)

            cv.imshow('Hand Tracking', cv.cvtColor(frame, cv.COLOR_RGB2BGR))
            key = cv.waitKey(10) & 0xFF

            if key == 27:  # Press 'ESC' to quit
                break
            elif key == ord('a'):  # Press 'A' to start recording
                recording = True
                session_data = []
                print("Recording started...")
            elif key == ord('s'):  # Press 'S' to stop recording
                recording = False
                print("Recording stopped.")
            elif key == ord('d'):  # Press 'D' to save recording
                recording = False
                label = input("Enter the label for this recording: ")
                for data in session_data:
                    data[0] = label
                    writer.writerow(data)
                print(f"Data saved with label: {label}")
                session_data = []

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
