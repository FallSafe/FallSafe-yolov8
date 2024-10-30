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
from email.mime.base import MIMEBase
from email import encoders
import cv2
import glob
import sys
import io
from dotenv import load_dotenv
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
        """Open a file dialog to select a video or image file."""

        for root, dirs, files in os.walk(self.save_dir):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                if os.path.isdir(dir_path):
                    for item in os.listdir(dir_path):
                        item_path = os.path.join(dir_path, item)
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                            print(f"Deleted file: {item_path}")
                    os.rmdir(dir_path)
                    print(f"Deleted directory: {dir_path}")
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")

        self.selected_file = filedialog.askopenfilename()

        if self.selected_file:
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, f"Selected file: {self.selected_file}\n")
            if self.selected_file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.dng', '.mpo', '.tif', '.tiff', '.webp', '.pfm', '.heic')):
                self.isImage = True
            elif self.selected_file.lower().endswith(('.asf', '.avi', '.gif', '.m4v', '.mkv', '.mov', '.mp4', '.mpeg', '.mpg', '.ts', '.wmv', '.webm')):
                self.isVideo = True
            else:
                self.fall_status_label.config(text="Invalid File Format : Pick again", style="FallDetected.TLabel")
                return False
            self.fall_status_label.config(text="Select 'Start Processing' to analyze the file", style="Select.TLabel")
            self.start_button.config(state=tk.NORMAL)
        
        return True
    
    def get_filename(self):
        match = re.search(r".*[\\/](.+)\.[^.]+$", self.selected_file)

        if match:
            return match.group(1)
        else:
            return None

    def validate_email(self, email):
        """Validate the email format."""
        email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
        return re.match(email_regex, email) is not None

    def update_gui(self, message):
        """Update the GUI text area with the given message."""
        self.output_text.insert(tk.END, message + '\n')
        self.output_text.see(tk.END)

    def start_processing(self):
        """Start processing the selected file in a separate thread."""
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
        if threading.Thread(target=self.run_yolo_command).start():
            self.fall_status_label.config(text="Processing Completed", style="Processing.TLabel")
            self.root.after(2000, self.root.destroy)

    def run_yolo_command(self):
        """Run the YOLO command to process the selected file, optimized for faster execution."""
        model_path = "model/model.pt"
        
        # Adjust command based on whether input is image or video
        common_params = f"model={model_path} source={self.selected_file} conf={CONFIDENCE_THRESHOLD} save=True project={self.save_dir} name=output device=0 workers=8 batch=32"
        
        if self.isImage:
            command = f"yolo predict {common_params}"
        elif self.isVideo:
            command = f"yolo predict {common_params} half=True format=mp4"
            
        else:
            self.update_gui("Error: Invalid file format for processing.")
            return False
        
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

        # Start the subprocess and handle output
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, shell=True, encoding='utf-8')
        except Exception as e:
            self.update_gui(f"Error running YOLO command: {e}")
            return False

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        with open(output_file_path, "w", encoding='utf-8') as output_file:
            for line in process.stdout:
                output_file.write(line)
                self.update_gui(line.strip())
                self.process_yolo_output(line)
        sys.stdout = sys.__stdout__ 

        self.write_frame_data()
        self.update_gui("Processing completed.")
        cv2.destroyAllWindows()
        
        return True

    def write_frame_data(self):
        """Write the frame data to a JSON file."""
        with open(frame_output_file, 'w', encoding='utf-8') as f:
            json.dump(self.frame_data, f, ensure_ascii=False, indent=4)  # Write frame data to JSON

    def process_yolo_output(self, line):
        """Extract labels and frame data from the YOLO output."""

        if self.total_frames % 3 != 0:  # Skipping frames and processing
            self.total_frames += 1
            return

        score_match = re.search(r'(image|video) \d+/\d+ \(frame \d+/\d+\) .+?: \d+x\d+ (fall|nofall) (\d+\.\d+), (fall|nofall) (\d+\.\d+)', line)

        if score_match:
            primary_label = score_match.group(2)
            primary_score = float(score_match.group(3))

            frame_info = {
                "frame_number": self.total_frames,
                "label": primary_label,
                "score": primary_score
            }
            self.frame_data.append(frame_info)

            self.fall_buffer.append(primary_label)
            if len(self.fall_buffer) > 20:  # Keep only the last 20 frames
                self.fall_buffer.pop(0)

            # Determine label based on the scores
            fall_count = self.fall_buffer.count('fall')
            label = 'fall' if fall_count >= 12 else 'nofall'

            self.update_fall_status(label)

            # Send email notification if a fall is detected and not already sent
            if label == 'fall' and not self.fall_detected:
                print("Fall detected, sending email...")
                self.send_email_notification()
                self.fall_detected = True

            # Reset the fall detected state if no fall is detected
            if label == 'nofall':
                self.fall_detected = False

            self.total_frames += 1

            # Displaying the current frame
            self.display_frame(self.total_frames)

        else:
            print(f"Error: Unable to match line format: {line}")
        

    def update_fall_status(self, current_label):
        """Update the fall status based on the current label."""
        # Determine the fall status based on the current label
        if current_label == 'fall':
            self.fall_status_label.config(text="Fall Detected!", style="FallDetected.TLabel")
        elif current_label == 'nofall':
            self.fall_status_label.config(text="No Fall Detected", style="NoFallDetected.TLabel")
        else:
            self.fall_status_label.config(text="Eror Encountered", style="Processing.TLabel")


    def display_frame(self, frame_number):
        """Display a specific frame from the video file with the window size same as the image size."""
        cap = cv2.VideoCapture(self.selected_file)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if ret:
                cv2.resizeWindow("Frame", frame.shape[1], frame.shape[0])
                cv2.imshow("Frame", frame)  # Display the frame
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Wait for a key press, exit if 'q' is pressed
                    cv2.destroyAllWindows()
        cap.release()

    def send_email_notification(self):
        """Send an email notification when a fall is detected."""

        output_dir = os.path.join(self.save_dir, "output")
        original_video_path = os.path.join(output_dir, f"{self.filename}.avi")
        new_video_path = os.path.join(output_dir, f"{self.filename}.mp4")
        if os.path.exists(original_video_path):     
            os.rename(original_video_path, new_video_path)
        else:
            self.update_gui("Error: Output video not found.")
            return False

        sender_email = os.getenv('SENDER_EMAIL')
        sender_password = os.getenv('SENDER_PASSWORD')
        recipient_email = self.receiver_email.get()

        subject = "Fall Detected Alert"
        body = "A fall has been detected. Please check the attached file."
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject

        # Attach the email body
        msg.attach(MIMEText(body, 'plain'))
        # Attach the frame image if available
        if os.path.exists(f'{self.save_dir}/output'):
            try:
                for root, dirs, files in os.walk(f'{self.save_dir}/output'):
                    for file in files:
                        file_path = os.path.join(root, file)
                        attachment = open(file_path, "rb")
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(attachment.read())
                        encoders.encode_base64(part)
                        part.add_header('Content-Disposition', f'attachment; filename={file}')
                        msg.attach(part)
                attachment.close()
            except Exception as e:
                self.update_gui(f"Error attaching image: {e}")
                return

        try:
            # Set up the SMTP server and send the email
            smtp_server = os.getenv('SMTP_SERVER')
            port = 587
            if smtp_server and port:
                try:
                    with smtplib.SMTP(smtp_server, port) as server:
                        server.starttls()
                        server.login(sender_email, sender_password)
                        server.sendmail(sender_email, recipient_email, msg.as_string())
                        print(f"Email sent to {recipient_email}")
                        self.update_gui("Email sent successfully!")
                except Exception as e:
                    self.update_gui(f"Error sending email: {e}")  # Update the GUI with error message

            else:
                self.update_gui("Error: SMTP server or port not configured.")  # Error handling for missing config
        except smtplib.SMTPAuthenticationError as e:
            self.update_gui(f"Error sending email: Authentication failed. Please check your email credentials.")  # Error handling for authentication
        except smtplib.SMTPException as e:
            self.update_gui(f"Error sending email: {e}")  # Error handling for SMTP exceptions
        except Exception as e:
            self.update_gui(f"Error sending email: {e}")  # Error handling for other exceptions

if __name__ == "__main__":
    root = tk.Tk()
    app = FallDetectionApp(root)
    root.mainloop()
