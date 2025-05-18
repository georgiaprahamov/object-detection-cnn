from ultralytics import YOLO
import cv2
import numpy as np


class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize the YOLO object detector

        Args:
            model_path: Path to the YOLO model weights
        """
        self.model = YOLO(model_path)

    def detect(self, image_path, conf_threshold=0.25):
        """
        Detect objects in an image

        Args:
            image_path: Path to the input image
            conf_threshold: Confidence threshold for detections

        Returns:
            Processed image with bounding boxes
        """
        # Read the image
        img = cv2.imread(image_path)

        # Run YOLOv8 inference
        results = self.model(img, conf=conf_threshold)

        # Process the results
        for result in results:
            boxes = result.boxes.cpu().numpy()

            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Get confidence score
                conf = float(box.conf[0])

                # Get class ID and name
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Create label with class name and confidence
                label = f"{cls_name}: {conf:.2f}"

                # Calculate text size and position
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), (0, 255, 0), -1)

                # Add text
                cv2.putText(img, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        return img