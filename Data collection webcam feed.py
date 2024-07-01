import cv2
import mediapipe as mp
import numpy as np
import os
from tkinter import Tk, Label, Entry, Button, StringVar, Canvas, Frame, messagebox, LEFT, RIGHT, Y, BOTH
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

# Set up video capture
cap = cv2.VideoCapture(0)

def update_frame():
    global is_finished
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                if is_recording:
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append([lm.x, lm.y, lm.z])
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

cap.release()
cv2.destroyAllWindows()
