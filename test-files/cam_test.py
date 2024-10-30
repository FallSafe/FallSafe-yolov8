import cv2

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open camera")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
