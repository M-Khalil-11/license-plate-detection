import cv2
from ultralytics import YOLO

class LicensePlateDetector:
    def __init__(self, plate_model_path, ocr_model_path):
        """Initialize the license plate detector with model paths."""
        self.plate_model = YOLO(plate_model_path)
        self.ocr_model = YOLO(ocr_model_path)
        self.char_map = self._create_char_map()
        
    def _create_char_map(self):
        """Create character mapping dictionary for OCR."""
        char_map = {}
        for k, v in self.ocr_model.names.items():
            if int(v) < 10:  # Numbers 0-9
                char_map[k] = v
            else:  # Letters A-Z
                char_map[k] = chr(int(v) - 10 + ord('A'))
        return char_map

    def process_plate_text(self, detections):
        """Process OCR detections and return the license plate text."""
        chars = []
        for det in detections[0].boxes:
            x_center = (det.xyxy[0][0] + det.xyxy[0][2])/2
            class_id = int(det.cls)
            char = self.char_map[class_id]
            chars.append((x_center, char))
        
        chars.sort(key=lambda x: x[0])
        return ''.join(char for _, char in chars)
