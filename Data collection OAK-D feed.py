import cv2
import mediapipe as mp
import numpy as np
import os
from tkinter import Tk, Label, Entry, Button, StringVar, Canvas, Frame, messagebox, LEFT, RIGHT, Y, BOTH
from tkinter import ttk
from PIL import Image, ImageTk
import depthai as dai

# Initialize Mediapipe Holistic model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
mp_drawing = mp.solutions.drawing_utils

# Set up directories for saving data
data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Global variables for controlling the recording process
is_recording = False
is_finished = False
label = ''
recorded_data = []
session_words = set()


# Function to save the recorded data
def save_label():
    global label
    label = label_var.get()
    if not label:
        messagebox.showerror("Error", "Please enter a label before saving.")
        return
    label_dir = os.path.join(data_dir, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    save_data(label_dir)
    label_var.set('')  # Clear the label input field after saving
    session_words.add(label)
    update_treeview()


def save_data(label_dir):
    global recorded_data
    for idx, data in enumerate(recorded_data):
        np.save(os.path.join(label_dir, f'frame_{idx:03d}.npy'), data)
    recorded_data = []
    print(f"Data saved for label: {label}")


def update_treeview():
    treeview.delete(*treeview.get_children())
    for word in sorted(session_words):
        treeview.insert('', 'end', text=word)
    for word in sorted(os.listdir(data_dir)):
        if word not in session_words:
            treeview.insert('', 'end', text=word)


# Tkinter window for label input
root = Tk()
root.title("Sign Language Label Input")

# Create a frame for video and controls
frame = Frame(root)
frame.pack(side=LEFT)

# Canvas for video feed
canvas = Canvas(frame, width=640, height=480)
canvas.pack()

# Create a frame for label input and save button
input_frame = Frame(frame)
input_frame.pack()

label_var = StringVar()

label_label = Label(input_frame, text="Enter label:")
label_label.pack()
label_entry = Entry(input_frame, textvariable=label_var, state='disabled')
label_entry.pack()
save_button = Button(input_frame, text="Save", command=save_label, state='disabled')
save_button.pack()

# Create a Treeview for tracking words
tree_frame = Frame(root)
tree_frame.pack(side=RIGHT, fill=Y)
treeview = ttk.Treeview(tree_frame)
treeview.pack(fill=BOTH, expand=True)


def handle_keypress(event):
    global is_recording, is_finished
    key = event.char
    if key == 'A' and not is_recording:
        is_recording = True
        is_finished = False
        print("Recording started")
    elif key == 'S' and is_recording:
        is_recording = False
        is_finished = True
        label_entry.config(state='normal')
        save_button.config(state='normal')
        print("Recording finished")


# Set up DepthAI pipeline for OAK-D camera
pipeline = dai.Pipeline()
cam_rgb = pipeline.createColorCamera()
xout_video = pipeline.createXLinkOut()

xout_video.setStreamName("video")

cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

cam_rgb.video.link(xout_video.input)

# Connect to device and start pipeline
device = dai.Device(pipeline)
video_queue = device.getOutputQueue(name="video", maxSize=30, blocking=False)


def update_frame():
    global is_finished
    video_frame = video_queue.get()
    frame = video_frame.getCvFrame()

    # Resize the frame to 640x480
    frame = cv2.resize(frame, (640, 480))

    # Convert the BGR frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = holistic.process(frame_rgb)

    if result.pose_landmarks:
        pose_landmarks = result.pose_landmarks.landmark
        # Draw specific pose landmarks: shoulders, elbows, wrists
        landmarks_to_draw = [
            mp_holistic.PoseLandmark.LEFT_SHOULDER,
            mp_holistic.PoseLandmark.RIGHT_SHOULDER,
            mp_holistic.PoseLandmark.LEFT_ELBOW,
            mp_holistic.PoseLandmark.RIGHT_ELBOW,
            mp_holistic.PoseLandmark.LEFT_WRIST,
            mp_holistic.PoseLandmark.RIGHT_WRIST
        ]
        for landmark in landmarks_to_draw:
            landmark_coords = pose_landmarks[landmark]
            cx, cy = int(landmark_coords.x * frame.shape[1]), int(landmark_coords.y * frame.shape[0])
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            if is_recording:
                recorded_data.append([landmark_coords.x, landmark_coords.y, landmark_coords.z])

    if result.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if is_recording:
            landmarks = [[lm.x, lm.y, lm.z] for lm in result.left_hand_landmarks.landmark]
            recorded_data.append(landmarks)

    if result.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if is_recording:
            landmarks = [[lm.x, lm.y, lm.z] for lm in result.right_hand_landmarks.landmark]
            recorded_data.append(landmarks)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    canvas.create_image(0, 0, anchor='nw', image=imgtk)
    canvas.imgtk = imgtk  # Keep a reference to avoid garbage collection

    root.after(10, update_frame)


root.bind('<Key>', handle_keypress)
root.after(10, update_frame)

# Initialize Treeview with existing data
update_treeview()

root.mainloop()

device.close()
cv2.destroyAllWindows()
