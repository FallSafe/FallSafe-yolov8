import smtplib
import os
import re
import json
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import cv2
import glob
import sys
import io
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip

load_dotenv()

output_file_path = "output/classification_output.txt"
frame_output_file = "output/frame_output.json"
CONFIDENCE_THRESHOLD = 0.5

class FallDetectionApp:
    def __init__(self, root):
        self.root = root
        self.selected_file = None
        self.frame_data = []
        self.fall_count = 0
        self.total_frames = 0
        self.isImage = True
        self.isVideo = True
        self.save_dir = "output"
        self.filename = "junk"
        self.fall_buffer = []
        self.fall_detected = False

        self.setup_gui()

    def setup_gui(self):
        """Initialize the GUI components for user interaction."""
        self.root.title("Fall Detection System")
        main_frame = ttk.Frame(self.root)
        main_frame.pack(padx=10, pady=10, fill="both", expand=True)

        ttk.Button(main_frame, text="Select File", command=self.select_file).grid(row=0, column=0, padx=10, pady=10)
        self.start_button = ttk.Button(main_frame, text="Start Processing", command=self.start_processing, state=tk.DISABLED)
        self.start_button.grid(row=0, column=1, padx=10, pady=10)

        ttk.Label(main_frame, text="Recipient Email:").grid(row=1, column=0, padx=10, pady=5)
        self.receiver_email = ttk.Entry(main_frame, width=30)
        self.receiver_email.grid(row=1, column=1, padx=10, pady=5)

        self.output_text = tk.Text(main_frame, height=20, width=70)
        self.output_text.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

        self.fall_status_label = ttk.Label(main_frame, text="Select a file to start", style="Select.TLabel")
        self.fall_status_label.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

        style = ttk.Style()
        style.configure("FallDetected.TLabel", foreground="red")
        style.configure("NoFallDetected.TLabel", foreground="green")
        style.configure("Select.TLabel", foreground="blue")
        style.configure("Processing.TLabel", foreground="orange")

    def select_file(self):
        """Open file dialog to select an image or video file for processing."""
        directory = "output"
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Clear existing files in output folder
        for root, dirs, files in os.walk(self.save_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                try:
                    os.rmdir(dir_path)  # Remove empty directories
                    print(f"Deleted directory: {dir_path}")
                except Exception as e:
                    print(f"Error removing directory {dir_path}: {e}")

        self.selected_file = filedialog.askopenfilename()

        if self.selected_file:
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, f"Selected file: {self.selected_file}\n")
            if self.selected_file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.dng', '.mpo', '.tif', '.tiff', '.webp', '.pfm', '.heic')): 
                self.isImage = True
                self.isVideo = False
            elif self.selected_file.lower().endswith(('.asf', '.avi', '.gif', '.m4v', '.mkv', '.mov', '.mp4', '.mpeg', '.mpg', '.ts', '.wmv', '.webm')):
                self.isVideo = True
                self.isImage = False
            else:
                self.fall_status_label.config(text="Invalid File Format : Pick again", style="FallDetected.TLabel")
                return False
            self.fall_status_label.config(text="Select 'Start Processing' to analyze the file", style="Select.TLabel")
            self.start_button.config(state=tk.NORMAL)
        
        return True
    
    def get_filename(self):
        """Extract filename without extension from the selected file path."""
        if self.selected_file is None:
            return None  # or raise ValueError("No file selected.")

        match = re.search(r".*[\\/](.+)\.[^.]+$", self.selected_file)
        return match.group(1) if match else None

    def validate_email(self, email):
        """Validate email format using regex."""
        email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA0-9-]+\.[a-zA-Z0-9-.]+$'
        return re.match(email_regex, email) is not None

    def update_gui(self, message):
        """Display message in GUI output text area."""
        self.root.after(0, self._update_text, message)

    def _update_text(self, message):
        """Actually update the text in the text widget."""
        self.output_text.insert(tk.END, message + '\n')
        self.output_text.see(tk.END)

    def start_processing(self):
        """Begin processing the selected file and manage GUI updates."""
        files = glob.glob(os.path.join(self.save_dir , '*'))
        for file in files:
            try:
                os.remove(file)
                print(f"Deleted: {file}")
            except Exception as e:
                print(f"Error deleting {file}: {e}")

        email = self.receiver_email.get()
        if not email or not self.validate_email(email):
            self.update_gui("Error: Please enter a valid recipient email before starting processing.")
            return
        
        self.fall_status_label.config(text="Processing.....", style="Processing.TLabel")
        self.start_button.config(state=tk.DISABLED)
        self.update_gui("Processing started...")

        self.filename = self.get_filename()
        threading.Thread(target=self.convert_video_to_lowerfps, daemon=True).start()

    def convert_video_to_lowerfps(self):
        """Convert video to 15 FPS and save it using moviepy."""
        input_video = self.selected_file
        output_video = os.path.join(self.save_dir, f"converted_{self.filename}.mp4")
        
        try:
            # Load video and set FPS to 15
            clip = VideoFileClip(input_video)
            clip = clip.set_fps(30)
            clip.write_videofile(output_video, codec='libx264', audio_codec='aac', threads=4)

            self.update_gui(f"Video converted to 30 FPS and saved as {output_video}")
            self.process_video(output_video)
        except Exception as e:
            self.update_gui(f"Error converting video: {e}")
            return

    def process_video(self, video_path):
        """Run YOLO model prediction command to detect falls in the converted video."""
        model_path = "model/best.pt"
        
        common_params = f"model={model_path} source={video_path} conf={CONFIDENCE_THRESHOLD} save=True project={self.save_dir} name=output device=0 workers=8 batch=32"
        
        command = f"yolo predict {common_params} stream_buffer=False"

        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

        # Run YOLO command and capture output
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, shell=True, encoding='utf-8')
        except Exception as e:
            self.update_gui(f"Error running YOLO command: {e}")
            return False

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        with open(output_file_path, "w", encoding='utf-8') as output_file:
            if process.stdout is not None:  # Check if stdout is not None
                for line in process.stdout:
                    output_file.write(line)
                    self.update_gui(line.strip())
            else:
                self.update_gui("Error: No output from the process.")

        self.update_gui(f"Processing completed for {self.selected_file}.")
        self.fall_status_label.config(text="Processing completed", style="Select.TLabel")

    def process_yolo_output(self, line):
        """Process YOLOv8 output to detect falls and handle email alerts."""
        if "Class" in line and "confidence" in line:
            match = re.search(r"Class: (.*?), Confidence: (\d+\.\d+)", line)
            if not match:
                return

            primary_label = match.group(1)
            primary_score = float(match.group(2))

            if primary_label == "fall" and primary_score > CONFIDENCE_THRESHOLD:
                self.fall_count += 1
                self.fall_detected = True
                self.update_gui(f"Fall detected in frame {self.total_frames} with confidence: {primary_score}")
                self.send_email_alert(primary_label, primary_score)

    def send_email_alert(self, label, confidence_score):
        """Send an email notification when a fall is detected."""
        try:
            sender_email = os.getenv('SENDER_EMAIL')
            sender_password = os.getenv('SENDER_PASSWORD')
            recipient_email = self.receiver_email.get()

            subject = "Fall Detection Alert"
            body = f"A fall was detected with a confidence score of {confidence_score:.2f}."
            if sender_email is None or recipient_email is None:
                raise ValueError("Sender email and recipient email must be provided.")

            if sender_password is None:
                raise ValueError("Sender password must be provided.")

            message = MIMEMultipart()
            message["From"] = sender_email
            message["To"] = recipient_email
            message["Subject"] = subject
            message.attach(MIMEText(body, "plain"))

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, recipient_email, message.as_string())
                self.update_gui(f"Alert sent to {recipient_email}.")

        except Exception as e:
            self.update_gui(f"Error sending email: {e}")

def run():
    """Run the Fall Detection application."""
    root = tk.Tk()
    app = FallDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    run()
