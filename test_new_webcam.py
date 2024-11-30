import cv2
import uuid
import json
import smtplib
import os
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from ultralytics import YOLO
from PIL import Image
import imageio

# Initialize the YOLO model
model = YOLO("model/model.pt")

# Open the video stream
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Initialize variables for fall frames and the start time
fall_frames = []
fall_start_time = None


def send_email_alert(label, confidence_score):
    """Send an email notification when a fall is detected."""
    try:
        sender_email = os.getenv("SENDER_EMAIL")
        sender_password = os.getenv("SENDER_PASSWORD")
        recipient_email = "recipient@example.com"

        if sender_email is None or recipient_email is None:
            raise ValueError("Sender email and recipient email must be provided.")

        if sender_password is None:
            raise ValueError("Sender password must be provided.")

        subject = "Fall Detection Alert"
        body = f"A fall was detected with a confidence score of {confidence_score:.2f}."
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = recipient_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, message.as_string())
            print(f"Alert sent to {recipient_email}.")

    except Exception as e:
        print(f"Error sending email: {e}")


def create_gif(frames, filename="fall_detection.gif"):
    """Create a GIF from the list of frames."""
    images = [Image.fromarray(frame) for frame in frames]
    imageio.mimsave(
        filename, images, duration=0.5
    )  # Adjust duration for timing between frames


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    frame_resized = cv2.resize(frame, (640, 640))

    results = model(frame_resized)

    predictions = []
    fall_detected = False
    if results:
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xywh
                confidences = result.boxes.conf
                class_ids = result.boxes.cls
                names = result.names

                for box, conf, cls in zip(boxes, confidences, class_ids):
                    print(f"Detected: {names[int(cls)]}, Confidence: {conf.item()}")
                    if conf >= 0.1:
                        class_name = names[int(cls)]

                        detection_id = str(uuid.uuid4())

                        prediction = {
                            "x": box[0].item(),
                            "y": box[1].item(),
                            "width": box[2].item(),
                            "height": box[3].item(),
                            "confidence": conf.item(),
                            "class": class_name,
                            "class_id": int(cls),
                            "detection_id": detection_id,
                        }

                        predictions.append(prediction)

                        if class_name == "fall":
                            fall_detected = True
                            send_email_alert(class_name, conf.item())

    output = {"predictions": predictions}
    output_json = json.dumps(output, indent=2)
    print(output_json)

    # Capture fall frames and maintain a 10-second window
    if fall_detected:
        if fall_start_time is None:
            fall_start_time = time.time()  # Start the 10-second window

        fall_frames.append(frame)  # Store the fall frame

    # Check if 10 seconds have passed since the first fall
    if fall_start_time and time.time() - fall_start_time >= 10:
        if fall_frames:
            # Create a GIF from fall frames
            create_gif(fall_frames)
            print("GIF created from fall frames.")

        # Reset the fall detection window
        fall_frames = []
        fall_start_time = None

    # Draw bounding boxes
    for result in results:
        if hasattr(result, "boxes") and result.boxes is not None:
            for box in result.boxes.xywh:
                x1, y1, w, h = map(int, box)
                x2, y2 = x1 + w, y1 + h
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the video feed
    cv2.imshow("Live Video Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
