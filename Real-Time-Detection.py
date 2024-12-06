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
import numpy as np

model = YOLO("model/model.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

fall_frames = []
fall_start_time = None


def send_email_alert(label, confidence_score):
    """
    Sends an email alert when a fall is detected.
    """
    try:
        sender_email = os.getenv("SENDER_EMAIL")
        sender_password = os.getenv("SENDER_PASSWORD")
        recipient_email = "recipient@example.com"

        if not sender_email or not recipient_email or not sender_password:
            raise ValueError(
                "Sender email, recipient email, and password must be provided."
            )

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
    """
    Creates a GIF from the frames of the fall detection event.
    """
    pil_frames = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]

    pil_frames[0].save(
        filename,
        save_all=True,
        append_images=pil_frames[1:],
        optimize=False,
        duration=500,
        loop=0,
    )
    print(f"GIF saved as {filename}.")


def process_predictions(results):
    """
    Processes the YOLO results and returns predictions in a structured format.
    """
    predictions = []
    fall_detected = False

    for box in results.boxes:
        x, y, w, h = box.xywh[0].tolist()
        conf = box.conf[0].item()
        cls = box.cls[0].item()
        class_name = results.names[int(cls)]

        print(f"Detected: {class_name}, Confidence: {conf}")

        if conf >= 0.1:
            detection_id = str(uuid.uuid4())
            prediction = {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "confidence": conf,
                "class": class_name,
                "class_id": int(cls),
                "detection_id": detection_id,
            }

            predictions.append(prediction)

            if class_name == "fall":
                fall_detected = True
                send_email_alert(class_name, conf)

    return predictions, fall_detected


def main():
    """
    Main function to run the fall detection system, handle video stream, and process frames.
    """
    fall_frames = []
    fall_start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        frame_resized = cv2.resize(frame, (640, 640))
        results = model(frame_resized)

        predictions, fall_detected = process_predictions(results)

        output = {"predictions": predictions}
        output_json = json.dumps(output, indent=2)
        print(output_json)

        if fall_detected:
            if fall_start_time is None:
                fall_start_time = time.time()

            fall_frames.append(frame)

        if fall_start_time and time.time() - fall_start_time >= 10:
            if fall_frames:
                create_gif(fall_frames)
                print("GIF created from fall frames.")

            fall_frames = []
            fall_start_time = None

        for result in results:
            if hasattr(result, 'boxes'):
                for box in result.boxes:
                    x, y, w, h = map(int, box.xywh[0].tolist())
                    x2, y2 = x + w, y + h
                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Live Video Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
