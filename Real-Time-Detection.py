import cv2
import threading
import time
from ultralytics import YOLO
from flask import Flask, render_template, Response, request, jsonify
from Email import send_email_alert

# Load YOLOv8 model
model = YOLO("model/model.pt")

# Video capture
cap = cv2.VideoCapture(0)

# Flask app
app = Flask(__name__)

fall_detected = False
fall_detected_lock = threading.Lock()
fall_detected_time = None
email_set = False  # Track whether the email has been set

def process_predictions(results, frame):
    global fall_detected, fall_detected_time
    if isinstance(results, list):
        results = results[0]  # Getting the first result if it's a list

    # Access the Probs object
    probs = results.probs  

    # Extract probabilities for 'fall' and 'nofall'
    fall_prob = probs.top1conf.item()  # Use .item() to get the value
    nofall_prob = probs.top5conf[1].item()  # Use .item() for the second class

    print("fall_prob =", fall_prob, "nofall_prob =", nofall_prob)

    if results.boxes:
        for box in results.boxes:
            if box.cls[0] == 0:  # Assuming 'fall' class has index 0
                with fall_detected_lock:
                    if not fall_detected:
                        fall_detected = True
                        fall_detected_time = time.time()

                        # Save the current frame as an image
                        frame_path = f"fall_frame_{int(time.time())}.jpg"
                        cv2.imwrite(frame_path, frame)

                        # Send email with frame attachment
                        send_email_alert(
                            label="Fall Detected!",
                            confidence_score=box.conf[0].item(),  # Use .item() for confidence score
                            receiver_email="arbaaz14122002@gmail.com",
                            frame_path=frame_path
                        )
    return fall_detected


def clear_fall_detection():
    global fall_detected, fall_detected_time
    # Reset detection if no fall is detected for more than 5 seconds
    if fall_detected_time and time.time() - fall_detected_time > 5:
        with fall_detected_lock:
            fall_detected = False
            fall_detected_time = None

def generate_frames():
    global email_set
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if email_set:  # Process only if email is set
            results = model(frame)
            fall_detected = process_predictions(results, frame)
            clear_fall_detection()  # Clear detection after 5 seconds
            if hasattr(results, 'plot'):
                frame = results.plot()  # Draws predictions on the frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', fall_detected=fall_detected, email_set=email_set)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/send_email', methods=['POST'])
def send_email():
    global email_set
    data = request.get_json()
    recipient = data.get('email', None)
    if recipient:
        email_set = True  # Set email flag to true
        return jsonify({"message": "Email saved successfully!"})
    return jsonify({"message": "Invalid email address"}), 400

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
