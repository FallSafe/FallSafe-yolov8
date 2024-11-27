import cv2
import uuid
import json
from ultralytics import YOLO

# Load the custom YOLOv8 model from 'model/model.pt' for inference
model = YOLO('model/model.pt')  # Use the YOLO class to load the model for inference

# OpenCV video capture (use 0 for webcam or IP camera stream URL for CCTV)
cap = cv2.VideoCapture(0)  # Change 0 to the camera source for CCTV if needed

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    # Read frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Resize the frame to match the input size of the model (assuming 640x640)
    frame_resized = cv2.resize(frame, (640, 640))

    # Perform inference on the frame
    results = model(frame_resized)  # Inference on the resized frame

    # Process the results
    predictions = []
    if results:  # Check if results are not empty
        for result in results:
            # Ensure there are detections in the result
            if result.boxes is not None and len(result.boxes) > 0:
                # Get the detection boxes and class information
                boxes = result.boxes.xywh
                confidences = result.boxes.conf
                class_ids = result.boxes.cls
                names = result.names

                # Process each detection
                for box, conf, cls in zip(boxes, confidences, class_ids):
                    print(f"Detected: {names[int(cls)]}, Confidence: {conf.item()}")  # Debugging line
                    if conf >= 0.1:  # Lowered confidence threshold from 0.2 to 0.1
                        class_name = names[int(cls)]  # Get class name from model labels

                        # Create a unique detection ID
                        detection_id = str(uuid.uuid4())

                        # Prepare the prediction dictionary
                        prediction = {
                            "x": box[0].item(),
                            "y": box[1].item(),
                            "width": box[2].item(),
                            "height": box[3].item(),
                            "confidence": conf.item(),
                            "class": class_name,
                            "class_id": int(cls),
                            "detection_id": detection_id
                        }

                        predictions.append(prediction)

    # Create the final JSON structure
    output = {
        "predictions": predictions
    }

    # Convert to JSON string
    output_json = json.dumps(output, indent=2)

    # Output the result (you can print it or write to a file)
    print(output_json)

    # Display the frame with bounding boxes (optional)
    for result in results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes.xywh:
                x1, y1, w, h = map(int, box)
                x2, y2 = x1 + w, y1 + h
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding boxes

    cv2.imshow('Live Video Feed', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
