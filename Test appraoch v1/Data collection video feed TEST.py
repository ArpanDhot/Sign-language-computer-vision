import cv2
import mediapipe as mp
import numpy as np
import os
import time
from tkinter import Tk, Label, Entry, Button, StringVar, Canvas, Frame, messagebox, LEFT, RIGHT, Y, BOTH, filedialog
from tkinter import ttk
from PIL import Image, ImageTk

# Initialize Mediapipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Set up directories for saving data
data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Global variables for controlling the recording process
label = ''
recorded_data = []
session_words = set()
video_path = None

def reset_to_start():
    start_time_var.set('')
    end_time_var.set('')
    label_var.set('')
    upload_button.config(state='normal')
    start_process_button.config(state='disabled')
    check_clip_button.config(state='disabled')
    label_entry.config(state='disabled')
    save_button.config(state='disabled')

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
    reset_to_start()

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

# Function to select a video file
def select_video_file():
    global video_path
    video_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    if video_path:
        check_clip_button.config(state='normal')

# Function to check the clip before processing
def check_clip():
    start_time = start_time_var.get()
    end_time = end_time_var.get()
    if not start_time or not end_time:
        messagebox.showerror("Error", "Please enter start and end times.")
        return

    start_minutes, start_seconds = map(int, start_time.split(':'))
    end_minutes, end_seconds = map(int, end_time.split(':'))

    start_time_sec = start_minutes * 60 + start_seconds
    end_time_sec = end_minutes * 60 + end_seconds

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time_sec * fps)
    end_frame = int(end_time_sec * fps)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for i in range(start_frame, min(end_frame, frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        canvas.create_image(0, 0, anchor='nw', image=imgtk)
        canvas.imgtk = imgtk  # Keep a reference to avoid garbage collection
        root.update_idletasks()
        root.update()
        time.sleep(0.1)  # Introduce a delay between frames (adjust the delay as needed)

    cap.release()
    start_process_button.config(state='normal')

# Function to process a video file
def process_video():
    global recorded_data
    start_time = start_time_var.get()
    end_time = end_time_var.get()
    if not start_time or not end_time:
        messagebox.showerror("Error", "Please enter start and end times.")
        return

    start_minutes, start_seconds = map(int, start_time.split(':'))
    end_minutes, end_seconds = map(int, end_time.split(':'))

    start_time_sec = start_minutes * 60 + start_seconds
    end_time_sec = end_minutes * 60 + end_seconds

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time_sec * fps)
    end_frame = int(end_time_sec * fps)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for i in range(start_frame, min(end_frame, frame_count)):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
                recorded_data.append(landmarks)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        canvas.create_image(0, 0, anchor='nw', image=imgtk)
        canvas.imgtk = imgtk  # Keep a reference to avoid garbage collection
        root.update_idletasks()
        root.update()

    cap.release()
    label_entry.config(state='normal')
    save_button.config(state='normal')
    print("Video processing finished")

# Tkinter window for label input
root = Tk()
root.title("Sign Language Label Input")

# Create a frame for video and controls
frame = Frame(root)
frame.pack(side=LEFT)

# Canvas for video feed
canvas = Canvas(frame, width=640, height=480)
canvas.pack()

# Create a frame for start/end times, label input and buttons
input_frame = Frame(frame)
input_frame.pack()

start_time_var = StringVar()
end_time_var = StringVar()
label_var = StringVar()

start_time_label = Label(input_frame, text="Start time (mm:ss):")
start_time_label.pack()
start_time_entry = Entry(input_frame, textvariable=start_time_var)
start_time_entry.pack()

end_time_label = Label(input_frame, text="End time (mm:ss):")
end_time_label.pack()
end_time_entry = Entry(input_frame, textvariable=end_time_var)
end_time_entry.pack()

upload_button = Button(input_frame, text="Upload Video", command=select_video_file)
upload_button.pack()

check_clip_button = Button(input_frame, text="Check Clip", command=check_clip, state='disabled')
check_clip_button.pack()

start_process_button = Button(input_frame, text="Start Process", command=process_video, state='disabled')
start_process_button.pack()

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

# Initialize Treeview with existing data
update_treeview()

root.mainloop()
