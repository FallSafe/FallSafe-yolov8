from ultralytics import YOLO
import time
import cv2

# Load a model
model = YOLO("model/model.pt")  # pretrained YOLOv11n model

# Run batched inference on a list of images
results = model(source=0, stream=False, half=True)  # return a generator of Results objects

# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    detected_classes = result.names  # Get class names from model

    # Ensure boxes and probs are not None
    if boxes is not None and probs is not None:
        # Loop over the predictions and check if 'fall' is detected
        for i, prob in enumerate(probs):  # Iterate over probs with index
            cls = boxes.cls[i]  # Get class for the current bounding box
            if detected_classes[int(cls)] == 'fall' and prob > 0.5:  # Threshold for confidence (0.5 can be adjusted)
                timestamp = time.strftime("%Y%m%d_%H%M%S")  # Get a unique timestamp for the filename
                filename = f"fall_frame_{timestamp}.jpg"  # Create a unique filename

                # Save the frame with the unique filename
                result.save(filename=filename)
                print(f"Fall detected. Frame saved as {filename}")
                
                # Show the frame (you can use OpenCV to display)
                frame = result.ims[0]  # Get the first image in the batch
                if frame is not None:
                    cv2.imshow("Fall Detected", frame)
                    cv2.waitKey(1)  # Delay to allow OpenCV to update the window
                else:
                    print("No frame to display.")
            
cv2.destroyAllWindows()  # Close all OpenCV windows when done
