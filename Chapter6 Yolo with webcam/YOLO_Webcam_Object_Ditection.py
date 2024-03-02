from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0)                                           # for webcam
cap.set(3,1280)                                        # sets the width of the video frames to 1280 pixels,
cap.set(4,720)                                         # sets the height of the video frames to 720 pixels.

model = YOLO("../Yolo-Weight/yolov8n.pt")                           # calling the yolov8n

classNames =["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
            "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"]



while True:
    succes, img = cap.read()                                       # Capture video frame
    results = model(img,stream=True)                               # Perform object detection on the video frame

    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding Box
            x1,y1,x2,y2 = box.xyxy[0]                              # Extract the coordinates of the bounding box
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)          # Convert the floating-point coordinates to integer coordinates
            w, h = x2-x1,y2-y1                                     # Calculate the width and height of the bounding box
            cvzone.cornerRect(img,(x1,y1,w,h))                # Draw a corner rectangle around the bounding box

            # Class name
            cls =int(box.cls[0])                                    # Get the class name of the detected object
            currentClass = classNames[cls]
            cvzone.putTextRect(img,f'{classNames[cls]} ',(max(0,x1),max(35,y1)),scale=2,thickness=2,offset=3)  # Add the class name to the video frame
            cvzone.cornerRect(img, (x1, y1, w, h),l=9)         # Draw a corner rectangle around the bounding box

    cv2.imshow("Image",img)                                 # Display the video frame with the added bounding boxes and class names
    cv2.waitKey(1)                                                  # Wait for a key press for 1 millisecond
