import torch
import cv2

# Load the YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5s", trust_repo=True)

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame from BGR (OpenCV) to RGB (YOLO expects RGB)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Inference
    results = model(img)

    # Render results on the original frame
    annotated_frame = results.render()[0]

    # Show the annotated frame
    cv2.imshow("YOLOv5 Detection", annotated_frame)

    # Press Esc to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
