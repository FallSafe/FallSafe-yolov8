import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import subprocess
import tkinter as tk
from tkinter import ttk
import threading
import re
import cv2
from PIL import Image, ImageTk
from dotenv import load_dotenv

load_dotenv()


output_file_path = "classification_output.txt"
fall_count = 0
total_frames = 0
last_label = "nofall"
is_processing = False

def run_yolo_command():
    global is_processing, fall_count, total_frames, last_label
    modelPath = "model\\model.pt"
    command = f"yolo classify predict model={modelPath} source=1 save=False"
    
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, shell=True)
    except Exception as e:
        update_gui(f"Error running YOLO command: {e}")
        is_processing = False
        return

    with open(output_file_path, "w") as output_file:
        for line in process.stdout:
            if not is_processing:
                break
            output_file.write(line)
            print(f"Debug: Raw output: {line.strip()}")
            update_gui(f"Debug: Raw output: {line.strip()}")

            score_match = re.search(r'(\w+)\s+([\d.]+)', line)
            if score_match:
                label, score = score_match.groups()
                score = float(score)
                print(f"Debug: Matched - Label: {label}, Score: {score}")  
                update_gui(f"Debug: Matched - Label: {label}, Score: {score}")  

                if label == 'fall' and score > 0.5:  # Threshold
                    current_label = 'fall'
                else:
                    current_label = 'nofall'

                print(f"Debug: Current label: {current_label}")  
                update_gui(f"Debug: Current label: {current_label}")  
                root.after(0, update_gui, f"Status: {current_label}, Score: {score:.2f}")
                root.after(0, update_fall_status, current_label)
                last_label = current_label

            if "Sending email notification" in line or "Email sent successfully!" in line:
                root.after(0, update_gui, line.strip())

    is_processing = False
    root.after(0, final_status_update)

def update_gui(text):
    output_text.insert(tk.END, text + "\n")
    output_text.see(tk.END)
    root.update_idletasks()

def update_fall_status(label):
    global fall_count, total_frames, last_label
    total_frames += 1
    last_label = label

    if label == "fall":
        fall_count += 1
        fall_status_label.config(text="Fall Detected!", fg="red")
        send_message()  # Sending email when fall is detected
    else:
        fall_status_label.config(text="No Fall Detected", fg="green")

def final_status_update():
    if last_label == "fall":
        fall_status_label.config(text="Fall Detected!", fg="red")
    else:
        fall_status_label.config(text="No Fall Detected", fg="green")
    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)

def send_message():
    subject = 'Fall Detected'
    body = 'Fall Detected!! Check the output file for more details.'

    msg = MIMEMultipart()
    msg['From'] = os.environ.get('SENDER_EMAIL', 'your_email@example.com')
    msg['To'] = receiver_email.get()
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    attach_file(msg, output_file_path)

    try:
        smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
        port = int(os.environ.get('PORT', 587))
        with smtplib.SMTP(smtp_server, port) as server:
            server.starttls()
            server.login(os.environ['SENDER_EMAIL'], os.environ['PASSWORD'])
            server.send_message(msg)
            print('Email sent successfully!')
            notification_label.config(text='Email sent successfully!', fg='green')
    except Exception as e:
        print(f'Error occurred while sending email: {e}')
        notification_label.config(text='Error sending email', fg='red')

def attach_file(msg, file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(file.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(file_path)}')
            msg.attach(part)

def validate_email(email):
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(email_regex, email) is not None

def start_processing():
    global is_processing, fall_count, total_frames
    if not is_processing:
        email = receiver_email.get()
        if not email or not validate_email(email):
            update_gui("Error: Please enter a valid recipient email before starting processing.")
            return
        
        is_processing = True
        start_button.config(state=tk.DISABLED)
        stop_button.config(state=tk.NORMAL)
        
        fall_count = 0
        total_frames = 0
        
        threading.Thread(target=run_yolo_command, daemon=True).start()

def stop_processing():
    global is_processing
    is_processing = False
    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)

def update_video():
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        video_label.config(image=photo)
        video_label.image = photo
    root.after(10, update_video)

def on_closing():
    cap.release()
    root.destroy()

root = tk.Tk()
root.title("Fall Detection System")
root.geometry("1000x800")

video_label = ttk.Label(root)
video_label.pack(padx=10, pady=10)

receiver_email_label = ttk.Label(root, text="Recipient Email:")
receiver_email_label.pack(pady=5)

receiver_email = ttk.Entry(root, width=30)
receiver_email.pack(pady=5)

# Create and pack the output text widget
output_text = tk.Text(root, wrap=tk.WORD, width=70, height=10)
output_text.pack(padx=10, pady=10)

# Label to show fall detection status
fall_status_label = ttk.Label(root, text="No Fall Detected", font=("Helvetica", 16))
fall_status_label.pack(pady=10)

# Label to show email sending status
notification_label = ttk.Label(root, text="", font=("Helvetica", 12))
notification_label.pack(pady=10)

# Create and pack the start and stop buttons
start_button = tk.Button(root, text="Start Processing", command=start_processing)
start_button.pack(pady=10)

stop_button = tk.Button(root, text="Stop Processing", command=stop_processing, state=tk.DISABLED)
stop_button.pack(pady=10)

# Initialize camera
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open camera")
    output_text.insert(tk.END, "Error: Could not open camera\n")

update_video()

root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()
