import os
import re
import threading
import tkinter as tk
from tkinter import ttk, filedialog
import glob
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip
from Whatsapp import send_whatsapp_alert
from Message import send_sms_alert
from Email import send_email_alert
from ultralytics import YOLO
from queue import Queue
import json
import uuid
load_dotenv()

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
        self.model = YOLO("model/model.pt")
        self.email_status_queue = Queue()

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

        ttk.Label(main_frame, text="Recipient Phone:").grid(row=2, column=0, padx=10, pady=5)
        self.receiver_phone = ttk.Entry(main_frame, width=30)
        self.receiver_phone.grid(row=2, column=1, padx=10, pady=5)

        self.output_text = tk.Text(main_frame, height=20, width=70)
        self.output_text.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

        self.fall_status_label = ttk.Label(main_frame, text="Select a file to start", style="Select.TLabel")
        self.fall_status_label.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

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
                    os.rmdir(dir_path)
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
        
        # Process differently based on file type
        if self.isVideo:
            threading.Thread(target=self.convert_video_to_lowerfps, daemon=True).start()
        else:  # Image processing
            threading.Thread(target=lambda: self.process_video(self.selected_file), daemon=True).start()

    def convert_video_to_lowerfps(self):
        """Convert video to 30 FPS and save it using moviepy2."""
        try:
            input_video = self.selected_file
            output_video = os.path.join(self.save_dir, f"converted_{self.filename}.mp4")
            
            self.root.after(0, lambda: self.update_gui("Starting video conversion..."))
            clip = VideoFileClip(input_video)
            clip = clip.set_fps(30)
            clip.write_videofile(output_video, codec='libx264', audio_codec='aac', threads=4)
            clip.close()

            self.root.after(0, lambda: self.update_gui("Video converted successfully. Starting processing..."))
            
            # Process the converted video
            self.process_video(output_video)

        except Exception as e:
            self.root.after(0, lambda: self.update_gui(f"Error in conversion: {str(e)}"))
            self.root.after(0, lambda: self.fall_status_label.config(
                text="Conversion Failed", 
                style="FallDetected.TLabel"
            ))
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))

    def process_video(self, video_path):
        """Process video using YOLO model with JSON output"""
        try:
            self.update_gui(f"Loading model and starting prediction...")
            
            # Verify video path exists
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            self.update_gui(f"Found video file at: {video_path}")

            # Verify model is loaded
            if self.model is None:
                raise ValueError("YOLO model not properly initialized")
            self.update_gui("Model verified")

            # Start prediction with detailed logging
            self.update_gui("Starting YOLO prediction...")
            results = self.model.predict(
                source=video_path,
                conf=CONFIDENCE_THRESHOLD,
                save=True,
                project=self.save_dir,
                name="output",
                stream=True,
                verbose=True  # Add verbose output
            )
            self.update_gui("YOLO prediction initialized")

            self.update_gui("Starting frame processing...")
            all_predictions = []
            frame_count = 0
            
            # Process each frame with logging
            for result in results:
                frame_count += 1
                self.update_gui(f"Processing frame {frame_count}")
                
                frame_predictions = self.process_frame_results(result)
                if frame_predictions:
                    all_predictions.extend(frame_predictions)
                    output = {"predictions": frame_predictions}
                    self.update_gui(f"Frame {frame_count} detections: {json.dumps(output, indent=2)}")
                    
                    # Check for falls and send alerts
                    for pred in frame_predictions:
                        if pred["class"] == "fall" and pred["confidence"] > CONFIDENCE_THRESHOLD:
                            self.fall_detected = True
                            self.update_gui(f"Fall detected in frame {frame_count} with confidence {pred['confidence']}")
                            self.send_alerts("fall", pred["confidence"])

            self.update_gui(f"Processed total of {frame_count} frames")
            self.update_gui("Video processing completed")
            
            # Update GUI status
            final_status = "Processing Complete - Falls Detected" if self.fall_detected else "Processing Complete - No Falls Detected"
            final_style = "FallDetected.TLabel" if self.fall_detected else "NoFallDetected.TLabel"
            
            self.root.after(0, lambda: self.fall_status_label.config(text=final_status, style=final_style))
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
            
            return {"predictions": all_predictions}

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.update_gui(f"Error processing video: {str(e)}")
            self.update_gui(f"Error details:\n{error_details}")
            self.root.after(0, lambda: self.fall_status_label.config(text="Processing Failed", style="FallDetected.TLabel"))
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
            return {"predictions": [], "error": str(e)}

    def process_frame_results(self, result):
        """Process single frame results into structured JSON format"""
        predictions = []
        
        if not hasattr(result, 'boxes'):
            return predictions

        for box in result.boxes:
            x, y, w, h = box.xywh[0].tolist()
            conf = box.conf[0].item()
            cls = box.cls[0].item()
            class_name = result.names[int(cls)]

            prediction = {
                "x": float(x),
                "y": float(y),
                "width": float(w),
                "height": float(h),
                "class": class_name,
                "confidence": float(conf),
                "detection_id": str(uuid.uuid4())
            }
            
            predictions.append(prediction)

        return predictions

    def send_alerts(self, label, confidence_score):
        """Send alerts with proper thread handling"""
        email_thread = threading.Thread(
            target=lambda q: q.put(send_email_alert(label, confidence_score)),
            args=(self.email_status_queue,),
            daemon=True
        )
        email_thread.start()

        # threading.Thread(
        #     target=lambda: send_sms_alert(label, confidence_score),
        #     daemon=True
        # ).start()
        
        # threading.Thread(
        #     target=lambda: send_whatsapp_alert(label, confidence_score),
        #     daemon=True
        # ).start()

        if not self.email_status_queue.empty():
            email_status = self.email_status_queue.get()
            self.update_gui(f"Email alert status: {email_status}")


def run():
    """Run the Fall Detection application."""
    root = tk.Tk()
    app = FallDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    run()
