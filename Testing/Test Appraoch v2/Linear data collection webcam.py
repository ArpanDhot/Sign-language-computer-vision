import csv
import copy
import argparse
from collections import deque
import cv2 as cv
import numpy as np
import mediapipe as mp
import itertools


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    return parser.parse_args()


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

    point_history = deque(maxlen=16)
    all_landmarks = []
    recording = False

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
                # Collect only the fingertip points
                fingertip_points.extend([landmark_list[i] for i in [4, 8, 12, 16, 20]])

                debug_image = draw_landmarks(debug_image, landmark_list)

            point_history.append(fingertip_points)
            debug_image = draw_point_history(debug_image, point_history)

            if recording and len(all_hand_landmarks) == 42:
                normalized_landmarks = pre_process_landmark(all_hand_landmarks)
                normalized_point_history = pre_process_landmark(fingertip_points)
                all_landmarks.append((normalized_landmarks, normalized_point_history))

        cv.imshow('Data Collection', debug_image)
        key = cv.waitKey(10)
        if key == 27:
            break
        if key == ord('a'):
            recording = True
            print("Recording started...")
        if key == ord('b'):
            recording = False
            print("Recording stopped.")
        if key == ord('d'):
            label = input("Enter label name: ")
            save_data(label, all_landmarks)
            all_landmarks = []

    cap.release()
    cv.destroyAllWindows()


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    return [[landmark.x * image_width, landmark.y * image_height]
            for landmark in landmarks.landmark]


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


def save_data(label, data):
    with open('landmarks.csv', 'a', newline="") as f:
        writer = csv.writer(f)
        for landmarks, point_history in data:
            writer.writerow([label, *landmarks])
    with open('point_history.csv', 'a', newline="") as f:
        writer = csv.writer(f)
        for landmarks, point_history in data:
            writer.writerow([label, *point_history])
    print(f"Data saved with label '{label}'.")


def draw_landmarks(image, landmark_list):
    for index, (x, y) in enumerate(landmark_list):
        cv.circle(image, (int(x), int(y)), 5, (255, 255, 255), -1)
        cv.circle(image, (int(x), int(y)), 5, (0, 0, 0), 1)
    return image


def draw_point_history(image, point_history):
    for index, points in enumerate(point_history):
        for (x, y) in points:
            if x != 0 and y != 0:
                cv.circle(image, (int(x), int(y)), 1 + int(index / 2), (152, 251, 152), 2)
    return image


if __name__ == '__main__':
    main()
