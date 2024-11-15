import cv2
import os
import smtplib
import numpy as np
import subprocess
import threading
from tkinter import *
from PIL import Image, ImageTk
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from queue import Queue

# Define constants
OUTPUT_FILE_PATH = "fall_detection_report.txt"
CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for YOLO
MODEL_PATH = "model/model.pt"
MAX_WORKERS = 4  # Number of threads for YOLO processing
FALL_DETECTION_FRAMES = 1  # Number of consecutive frames with a fall to trigger email

class FallDetectionApp:
    def __init__(self, root):
        self.root = root
        self.save_dir = "output"
        self.running = False
        self.source = 0
        self.queue = Queue()
        self.fall_frame_count = 0  # Count of consecutive fall frames
        self.email_sent = False  # Flag to prevent multiple emails
        self.setup_gui()

    def setup_gui(self):
        self.root.title("Fall Detection System")
        video_frame = Frame(self.root)
        video_frame.pack(pady=10)
        
        self.video_label = Label(video_frame, width=640, height=480)
        self.video_label.pack()
        self.set_empty_frame()

        status_frame = Frame(self.root)
        status_frame.pack(pady=10)
        self.fall_status_label = Label(status_frame, text="Waiting for detection...", fg="yellow", font=("Helvetica", 14))
        self.fall_status_label.pack()

        email_frame = Frame(self.root)
        email_frame.pack(pady=10)
        Label(email_frame, text="Recipient Email:").pack(side=LEFT)
        self.receiver_email = Entry(email_frame, width=30)
        self.receiver_email.pack(side=LEFT)
        self.receiver_email.insert(0, "")

        button_frame = Frame(self.root)
        button_frame.pack(pady=10)
        start_button = Button(button_frame, text="Start Processing", command=self.start_processing)
        start_button.pack(side=LEFT, padx=5)
        stop_button = Button(button_frame, text="Stop Processing", command=self.stop_processing)
        stop_button.pack(side=LEFT, padx=5)

        self.notification_label = Label(self.root, text="", fg="green")
        self.notification_label.pack(pady=10)

    def set_empty_frame(self):
        blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
        imgtk = self.convert_frame_to_image(blank_image)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def clear_output_dir(self):
        for root_dir, dirs, files in os.walk(self.save_dir):
            for dir_name in dirs:
                os.rmdir(os.path.join(root_dir, dir_name))
            for file_name in files:
                os.remove(os.path.join(root_dir, file_name))

    def run_yolo_command(self):
        self.clear_output_dir()
        command = f"yolo classify predict model={MODEL_PATH} source={self.source} conf={CONFIDENCE_THRESHOLD} save=False project={self.save_dir} name={self.save_dir} device=0 workers={MAX_WORKERS} batch=32 half=True"

        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, encoding='utf-8')
            with open(OUTPUT_FILE_PATH, "w", encoding='utf-8') as output_file:
                for line in process.stdout:
                    output_file.write(line)
                    self.update_gui(line.strip())
                    self.process_yolo_output(line)
            self.update_gui("Processing completed.")
        except Exception as e:
            self.update_gui(f"Error running YOLO command: {e}", fg="red")

    def process_video(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        frame_count = 0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            fall_detected = self.detect_fall(frame)
            self.queue.put((fall_detected, frame, frame_count))

            imgtk = self.convert_frame_to_image(frame)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        self.set_empty_frame()

    def start_processing(self):
        self.running = True
        self.email_sent = False
        self.fall_frame_count = 0
        threading.Thread(target=self.process_video).start()
        threading.Thread(target=self.update_fall_status).start()

    def stop_processing(self):
        self.running = False
        self.update_gui("Processing stopped", fg="orange")
        self.set_empty_frame()

    def send_message(self):
        threading.Thread(target=self._send_email_thread).start()

    def _send_email_thread(self):
        subject = 'Fall Detected'
        body = 'Fall Detected! Check the output file for more details.'

        msg = MIMEMultipart()
        msg['From'] = os.environ.get('SENDER_EMAIL')
        msg['To'] = self.receiver_email.get()
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        self.attach_file(msg, OUTPUT_FILE_PATH)

        try:
            smtp_server = os.environ.get('SMTP_SERVER')
            port = int(os.environ.get('SMTP_PORT', 587))
            with smtplib.SMTP(smtp_server, port) as server:
                server.starttls()
                server.login(os.environ['SENDER_EMAIL'], os.environ['SENDER_PASSWORD'])
                server.send_message(msg)
                self.update_gui('Email sent successfully!', fg='green')
        except Exception as e:
            self.update_gui(f'Error sending email: {e}', fg='red')

    def attach_file(self, msg, file_path):
        if os.path.exists(file_path):
            with open(file_path, "rb") as file:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(file.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(file_path)}')
                msg.attach(part)

    def update_fall_status(self):
        while self.running or not self.queue.empty():
            try:
                fall_detected, frame, frame_count = self.queue.get(timeout=0.1)
                if fall_detected:
                    self.fall_frame_count += 1
                    self.save_fall_frame(frame, frame_count)
                    self.fall_status_label.config(text="Fall Detected", fg="red")
                    # Trigger email if fall is detected for 20 consecutive frames
                    if self.fall_frame_count >= FALL_DETECTION_FRAMES and not self.email_sent:
                        self.send_message()
                        self.email_sent = True
                else:
                    self.fall_frame_count = 0
                    self.fall_status_label.config(text="No Fall Detected", fg="green")
            except:
                pass

    def convert_frame_to_image(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))

    def save_fall_frame(self, frame, frame_count):
        folder_path = self.save_dir
        os.makedirs(folder_path, exist_ok=True)
        frame_filename = os.path.join(folder_path, f"fall_frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)
        self.update_gui(f"Saved frame: {frame_filename}")

    def update_gui(self, message, fg="green"):
        self.notification_label.config(text=message, fg=fg)

    def process_yolo_output(self, line):
        pass

    def detect_fall(self, frame):
        return False

# Initialize the GUI application
root = Tk()
app = FallDetectionApp(root)
root.mainloop()
