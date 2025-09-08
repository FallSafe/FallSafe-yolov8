import cv2
import torch
import threading
import multiprocessing
import time
from ultralytics import YOLO
from flask import Flask, render_template, Response, request, jsonify
from Email import send_email_alert
# from Whatsapp import send_whatsapp_alert
# from Message import send_sms_alert
from icecream import ic
import os
import shutil

model_path = "model/model.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = YOLO(model_path)

app = Flask(__name__)

fall_detected = False
fall_detected_lock = threading.Lock()
fall_detected_time = None
alert_set = False
recipient = ""
tonumber = ""
confidence = 1.0

def process_predictions(results, frame):
    """
    Process the classification results. For each result, extract the top predicted class and
    confidence using the 'probs' attribute. If a 'fall' is detected with a confidence
    equal to or above the user-specified threshold, send an email alert.
    """
    global fall_detected, fall_detected_time

    if not results:
        print("No results returned by the model.")
        return fall_detected

    for r in results:
        probs = r.probs  
        if probs is None:
            print("No probability scores available in the result.")
            continue

        class_idx = probs.top1  
        pred_conf = probs.data[class_idx].item()  
        class_name = r.names[class_idx]  

        ic(f"Predicted Class: {class_name}, Confidence: {pred_conf:.2f}")
        if class_name == "fall":
            if pred_conf < confidence:
                ic(f"Skipping prediction because confidence {pred_conf:.2f} is below threshold {confidence:.2f}")
                continue
            fall_detected = True
            fall_detected_time = time.time()

            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            frame_path = os.path.join(output_dir, f"fall_frame_{fall_detected_time}.jpg")
            if not cv2.imwrite(frame_path, frame):
                print(f"Failed to save frame at {frame_path}")

            p_email = multiprocessing.Process(target=send_email_alert, kwargs={
                "label": "Fall Detected!",
                "confidence_score": pred_conf,
                "receiver_email": recipient,
                "frame_path": frame_path
            })
            p_email.start()
            p_email.join()

            # p_sms = multiprocessing.Process(target=send_sms_alert, args=(tonumber,))
            # p_whatsapp = multiprocessing.Process(target=send_whatsapp_alert, args=(tonumber,))
            # p_sms.start()
            # p_whatsapp.start()
            # p_sms.join()
            # p_whatsapp.join()

            ic(f"Fall detected with confidence: {pred_conf:.2f}")

        elif class_name == "nofall":
            fall_detected = False

    return fall_detected

def generate_frames():
    """
    Read frames from the camera, run inference if alerts are enabled, and yield JPEG-encoded frames
    for streaming via Flask.
    """
    global alert_set, confidence, cap
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame captured from camera.")
            break
        
        if alert_set:
            results = model.predict(source=frame, conf=confidence)
            process_predictions(results, frame)
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', fall_detected=fall_detected, alert_set=alert_set)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/send_details', methods=['POST'])
def send_alert():
    """
    Save the user provided email, phone, and confidence threshold and enable alert processing.
    """
    global alert_set, recipient, tonumber, confidence
    data = request.get_json()
    recipient = data.get('email')
    tonumber = data.get('phone')
    conf_value = data.get('conf')
    
    try:
        confidence = float(conf_value)
    except (TypeError, ValueError):
        return jsonify({"message": "Invalid confidence value"}), 400

    print("Received details:", data)
    print("Recipient:", recipient, "Phone:", tonumber, "Confidence:", confidence)
    
    if recipient and tonumber and confidence:
        alert_set = True
        return jsonify({"message": "Email and Phone saved successfully!"})
    return jsonify({"message": "Invalid details"}), 400

@app.route('/fall_status')
def updateFallStatus():
    return jsonify({"status": fall_detected})

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    output_dir = "output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    app.run(host='0.0.0.0', port=5000)
