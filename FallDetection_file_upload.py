import smtplib
import os
import re
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import cv2
import glob
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip
from PIL import Image, ImageTk
import multiprocessing
import imageio
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from ultralytics import YOLO
import json
import numpy as np

load_dotenv()

CONFIDENCE_THRESHOLD = 0.5

def create_output_directory():
    # Get the current date and time
    now = datetime.now()
    # Format the date and time for the folder name
    folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("output", folder_name)

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def convert_video_to_lowerfps(self):
    """Convert video to 30 FPS and save it using moviepy."""
    input_video = self.selected_file
    output_video = os.path.join(self.save_dir, f"converted_{self.filename}.mp4")

    # Check if the input video file is valid and exists
    if not input_video or not isinstance(input_video, str) or not os.path.exists(input_video):
        self.update_gui(f"Error: The video file does not exist or is invalid: {input_video}")
        return

    try:
        # Load video and set FPS to 30
        clip = VideoFileClip(input_video)
        clip = clip.set_fps(30)
        clip.write_videofile(output_video, codec='libx265', audio_codec='aac', threads=8)

        self.update_gui(f"Video converted to 30 FPS and saved as {output_video}")
    except Exception as e:
        self.update_gui(f"Error converting video: {e}")

def create_gif_from_frames(frames, gif_path):
    """Create a GIF from the list of frames."""
    imageio.mimsave(gif_path, frames, duration=0.5)  # Adjust duration as needed

def process_video_for_falls(input_video, output_gif, queue, overlap_threshold=0.5):
    """Process the video to detect falls and create a GIF."""
    fall_detected_frames = []  # List to hold frames where falls are detected
    predictions_list = []  # List to hold predictions for JSON output

    # Open a text file for logging
    log_file_path = "yolo_output_log.txt"
    with open(log_file_path, "w") as log_file:
        try:
            # Load the YOLO model
            model = YOLO("model/model.pt")  # Adjust the model path as needed

            # Open the video file using OpenCV
            cap = cv2.VideoCapture(input_video)

            if not cap.isOpened():
                queue.put("Error: Could not open video.")
                return

            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # Exit the loop if there are no more frames

                # Run YOLO on the current frame
                results = model(frame)  # Get results from the model

                # Log the results for debugging
                log_file.write(f"Processing frame {cap.get(cv2.CAP_PROP_POS_FRAMES)}\n")
                if results is None:
                    log_file.write("Warning: No results returned from YOLO model.\n")
                    continue

                # Process results
                for result in results:
                    if hasattr(result, 'boxes'):
                        boxes = result.boxes  # Get the bounding boxes
                        if boxes is None:
                            log_file.write("Warning: No boxes detected.\n")
                            continue
                        
                        # Apply Non-Maximum Suppression (NMS) based on the overlap threshold
                        boxes_data = []
                        for box in boxes:
                            class_id = int(box.cls[0])  # Get class ID
                            confidence = box.conf[0].item()  # Get confidence score
                            if confidence > 0.5:  # Adjust confidence threshold as needed
                                x, y, w, h = box.xywh[0].tolist()  # Get bounding box coordinates
                                boxes_data.append((x, y, w, h, confidence, class_id))

                        # Perform NMS
                        if boxes_data:
                            boxes_data = np.array(boxes_data)
                            x1 = boxes_data[:, 0] - boxes_data[:, 2] / 2
                            y1 = boxes_data[:, 1] - boxes_data[:, 3] / 2
                            x2 = boxes_data[:, 0] + boxes_data[:, 2] / 2
                            y2 = boxes_data[:, 1] + boxes_data[:, 3] / 2

                            # Calculate area
                            areas = (x2 - x1) * (y2 - y1)
                            indices = cv2.dnn.NMSBoxes(boxes_data[:, :4].tolist(), boxes_data[:, 4].tolist(), 0.5, overlap_threshold)

                            for index in indices:
                                box = boxes_data[index]  # Access the index directly
                                x, y, w, h, confidence, class_id = box
                                prediction = {
                                    "x": x,
                                    "y": y,
                                    "width": w,
                                    "height": h,
                                    "confidence": confidence,
                                    "class": result.names[class_id],  # Get class name
                                    "class_id": class_id,
                                }
                                predictions_list.append(prediction)

                                if class_id == 0:  # Assuming '0' is the class ID for 'fall'
                                    fall_detected_frames.append(frame)  # Add the frame to the list if a fall is detected

                # Optionally, draw bounding boxes on the frame for visualization
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle

                # Display the frame (optional)
                cv2.imshow("Live Video Feed", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()  # Release the video capture object
            cv2.destroyAllWindows()  # Close all OpenCV windows

            # Create JSON output
            output_json = {"predictions": predictions_list}
            print(json.dumps(output_json, indent=2))  # Print the JSON output

            # Create GIF from detected fall frames
            if fall_detected_frames:
                # Convert BGR frames to RGB for GIF creation
                fall_detected_frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in fall_detected_frames]
                imageio.mimsave(output_gif, fall_detected_frames_rgb, duration=0.5)  # Adjust duration as needed
                queue.put(f"GIF created and saved as {output_gif}")
            else:
                queue.put("No falls detected; GIF not created.")
        except Exception as e:
            queue.put(f"Error processing video: {e}")
            log_file.write(f"Error processing video: {e}\n")

def send_email_with_gif(gif_path, recipient_email):
    """Send an email with the GIF attached."""
    try:
        sender_email = os.getenv('SENDER_EMAIL')
        sender_password = os.getenv('SENDER_PASSWORD')

        if sender_email is None or sender_password is None or recipient_email is None:
            raise ValueError("Sender email, sender password, and recipient email must be provided.")

        subject = "Fall Detection Alert - GIF Attached"
        body = "A fall was detected. Please find the attached GIF showing the detected falls."

        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = recipient_email
        message["Subject"] = subject

        with open(gif_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(gif_path)}")
            message.attach(part)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(message)
            print(f"Alert sent to {recipient_email}.")
    except Exception as e:
        print(f"Error sending email: {e}")

class FallDetectionApp:
    def __init__(self, root):
        self.root = root
        self.selected_file = None
        self.frame_data = []
        self.fall_count = 0
        self.total_frames = 0
        self.isImage = True
        self.isVideo = True
        self.save_dir = create_output_directory()
        self.filename = "junk"
        self.fall_buffer = []
        self.fall_detected = False

        self.setup_gui()

    def setup_gui(self):
        """Initialize the GUI components for user interaction."""
        self.root.title("Fall Detection System")
        main_frame = ttk.Frame(self.root)
        main_frame.pack(padx=10, pady=10, fill="both", expand=True)

        ttk.Button(main_frame, text="Select File", command=self.select_file).grid(row=0, column=0, padx=10, pady=5)
        self.start_button = ttk.Button(main_frame, text="Start Processing", command=self.start_processing, state=tk.DISABLED)
        self.start_button.grid(row=0, column=1, padx=10, pady=5)

        ttk.Label(main_frame, text="Recipient Email:").grid(row=1, column=0, padx=10, pady=5)
        self.receiver_email = ttk.Entry(main_frame, width=30)
        self.receiver_email.grid(row=1, column=1, padx=10, pady=5)

        self.output_text = tk.Text(main_frame, height=25, width=70)
        self.output_text.grid(row=2, column=0, columnspan=2, padx=10, pady=5)

        self.image_canvas = tk.Canvas(main_frame, width=640, height=480)
        self.image_canvas.grid(row=3, column=0, columnspan=2, padx=10, pady=5)

        self.result_label = ttk.Label(main_frame, text="Detection Result: None", style="Select.TLabel")
        self.result_label.grid(row=4, column=0, columnspan=2, padx=10, pady=5)

        self.fall_status_label = ttk.Label(main_frame, text="Select a file to start", style="Select.TLabel")
        self.fall_status_label.grid(row=5, column=0, columnspan=2, padx=10, pady=5)

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
        files = glob.glob(os.path.join(self.save_dir, '*'))
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

        # Start processing in a separate thread
        if self.isVideo:
            input_video = self.selected_file
            output_gif = os.path.join(self.save_dir, f"falls_detected.gif")
            queue = multiprocessing.Queue()
            process = multiprocessing.Process(target=process_video_for_falls, args=(input_video, output_gif, queue))
            process.start()
            self.root.after(100, self.check_process, process, queue)  # Check the process periodically
        else:
            self.process_image(self.selected_file)  # Process image

    def check_process(self, process, queue):
        """Check if the process is still running and update the GUI."""
        output_gif = os.path.join(self.save_dir, f"falls_detected.gif")
        if process.is_alive():
            self.root.after(100, self.check_process, process, queue)  # Check again after 100ms
        else:
            # Process finished, get the result from the queue
            while not queue.empty():
                message = queue.get()
                self.update_gui(message)  # Update the GUI with the message
                if "GIF created" in message:
                    self.send_email_with_gif(output_gif, self.receiver_email.get())  # Send the GIF via email
            self.start_button.config(state=tk.NORMAL)  # Re-enable the start button

    def process_file(self):
        """Process the selected file (image or video) in a separate thread."""
        self.filename = self.get_filename()

        if self.isVideo:
            input_video = self.selected_file
            output_gif = os.path.join(self.save_dir, f"falls_detected.gif")
            queue = multiprocessing.Queue()
            process = multiprocessing.Process(target=process_video_for_falls, args=(input_video, output_gif, queue))
            process.start()
            self.root.after(100, self.check_process, process, queue)  # Check the process periodically
        else:
            self.process_image(self.selected_file)  # Process image

        self.update_gui("Processing completed.")
        self.fall_status_label.config(text="Processing completed", style="Select.TLabel")
        self.start_button.config(state=tk.NORMAL)  # Re-enable the start button

    def process_image(self, image_path):
        """Process the selected image for fall detection."""
        self.update_gui(f"Processing image: {image_path}")
        
        model_path = "model/model.pt"
        command = f"yolo predict model={model_path} source={image_path} conf={CONFIDENCE_THRESHOLD} save=True project={self.save_dir} name=output device=0"

        try:
            # Use subprocess.run to execute the command and wait for it to complete
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)

            # Check for errors
            if result.returncode != 0:
                self.update_gui(f"Error processing image: {result.stderr}")
                return

            # Process the output if needed
            self.update_gui(result.stdout)  # Display any output from the command

            # Check if the output files were created
            output_file_path = os.path.join(self.save_dir, "output")  # Adjust based on your output structure
            if os.path.exists(output_file_path):
                self.update_gui("Detection completed successfully.")
                self.handle_detection_results(output_file_path, image_path)  # Pass image_path for email
            else:
                self.update_gui("No output files found. Please check the YOLO command.")

        except Exception as e:
            self.update_gui(f"Error processing image: {e}")

        # Ensure the GUI reflects that processing is complete
        self.start_button.config(state=tk.NORMAL)  # Re-enable the start button

    def handle_detection_results(self, output_path, image_path):
        """Handle the results of the YOLO detection and send email with results."""
        # Check if the uploaded file is a video or an image
        if self.isVideo:
            output_gif = os.path.join(output_path, "falls_detected.gif")  # Adjust as needed
            if os.path.exists(output_gif):
                self.send_email_with_gif(output_gif, self.receiver_email.get())  # Send the GIF via email
                self.update_gui("GIF created and email sent.")
            else:
                self.update_gui("No GIF created from detection results.")
        else:
            # If it's an image, send the original image
            if os.path.exists(image_path):
                self.send_email_with_image(image_path, self.receiver_email.get())  # Send the image via email
                self.update_gui("Image sent via email.")
            else:
                self.update_gui("Original image not found; email not sent.")

    def send_email_with_image(self, image_path, recipient_email):
        """Send an email with the image attached."""
        try:
            sender_email = os.getenv('SENDER_EMAIL')
            sender_password = os.getenv('SENDER_PASSWORD')

            if sender_email is None or sender_password is None or recipient_email is None:
                raise ValueError("Sender email, sender password, and recipient email must be provided.")

            subject = "Fall Detection Alert - Image Attached"
            body = "A fall was detected. Please find the attached image showing the detected fall."

            message = MIMEMultipart()
            message["From"] = sender_email
            message["To"] = recipient_email
            message["Subject"] = subject
            message["Body"] = body

            with open(image_path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(image_path)}")
                message.attach(part)

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(sender_email, sender_password)
                server.send_message(message)
                print(f"Image alert sent to {recipient_email}.")
        except Exception as e:
            self.update_gui(f"Error sending image email: {e}")

    def display_detected_images(self):
        """Display the detected images in the UI."""
        detected_images_path = os.path.join(self.save_dir, "output")  # Adjust the path as necessary
        for image_file in glob.glob(os.path.join(detected_images_path, "*.jpg")):  # Assuming images are saved as .jpg
            img = cv2.imread(image_file)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                img = cv2.resize(img, (640, 480))  # Resize for display
                img = Image.fromarray(img)  # Ensure img is a valid PIL image
                self.imgtk = ImageTk.PhotoImage(img)  # Create PhotoImage from PIL image
                self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.imgtk)  # Display image in the canvas
                self.image_reference = self.imgtk  # Store reference in the class instance to avoid garbage collection
                self.output_text.insert(tk.END, "\n")  # Add a newline after each image

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
                self.update_detection_result("Fall Detected")  # Update the result label
            else:
                self.update_detection_result("No Fall Detected")  # Update the result label

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

    def update_detection_result(self, result):
        """Update the detection result label with the result of the detection."""
        self.result_label.config(text=f"Detection Result: {result}")

    def send_email_with_gif(self, gif_path, recipient_email):
        """Send an email with the GIF attached."""
        try:
            sender_email = os.getenv('SENDER_EMAIL')
            sender_password = os.getenv('SENDER_PASSWORD')

            if sender_email is None or sender_password is None or recipient_email is None:
                raise ValueError("Sender email, sender password, and recipient email must be provided.")

            subject = "Fall Detection Alert - GIF Attached"
            body = "A fall was detected. Please find the attached GIF showing the detected falls."

            message = MIMEMultipart()
            message["From"] = sender_email
            message["To"] = recipient_email
            message["Subject"] = subject

            with open(gif_path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(gif_path)}")
                message.attach(part)

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(sender_email, sender_password)
                server.send_message(message)
                print(f"Alert sent to {recipient_email}.")
        except Exception as e:
            self.update_gui(f"Error sending email: {e}")

def run():
    """Run the Fall Detection application."""
    root = tk.Tk()
    root.geometry("700x600")
    app = FallDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    run()
