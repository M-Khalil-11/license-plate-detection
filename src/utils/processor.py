import cv2
import os
import numpy as np
from .detector import PlateDetector
from .ocr import OCRModel
from .visualization import Visualizer
from .config import Config
class PlateProcessor:
    def __init__(self):
        self.plate_detector = PlateDetector(Config.PLATE_MODEL_PATH)
        self.ocr_model = OCRModel(Config.OCR_MODEL_PATH)
        self.visualizer = Visualizer()
        self.detected_plates = {}
        self.counter = []  
        self.car_movements = {}  
        self.entering_count = 0  
        self.exiting_count = 0  

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), "Error reading video file"
        
        # Get video properties
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize video writer
        if Config.SAVE_VIDEO:
            vid = video_path.split('/')[-1].split('.')[0]
            output_path = os.path.join(Config.OUTPUT_DIR, f"processed_video_{vid}.mp4")
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        # Define ROI and reference line
        area = [(0, h-200), 
                (0, h-300), 
                (w, h-300), 
                (w, h-200)]
        line_y = h - 250  
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Process frame
            display_frame = self.process_frame(frame, area, line_y)
            # Display car count
            display_frame = self.display_car_counts(display_frame)
            # Save frame if configured
            if Config.SAVE_VIDEO:
                out.write(display_frame)
            
            # Display results if configured
            if Config.DISPLAY_RESULTS:
                cv2.imshow("License Plate Detection & OCR", cv2.resize(display_frame, (1280, 720)))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        
        cap.release()
        if Config.SAVE_VIDEO:
            out.release()
        cv2.destroyAllWindows()
        
        return self.detected_plates

    def process_frame(self, frame, area, line_y):   
        # Create a copy of the frame for visualization
        display_frame = frame.copy()
        self.visualizer.draw_roi(display_frame, area)

        # Detect and track plates
        track_results = self.plate_detector.detect_and_track(frame)
        boxes = track_results[0].boxes.xyxy.int().cpu().tolist()
        track_ids = track_results[0].boxes.id.int().cpu().tolist()

        if boxes is not None:
            for box, track_id in zip(boxes, track_ids):
                # Calculate center point of detected plate
                cx = int(box[0] + box[2]) // 2
                cy = int(box[1] + box[3]) // 2

                # Check if plate is in ROI
                in_roi = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False) >= 0

                if in_roi:
                    if track_id not in self.counter:
                        # Process new plate
                        self.process_new_plate(frame, box, track_id)

                    # Track movement for direction detection
                    self.detect_direction(track_id, cy, line_y)

                # Draw detection
                text = self.detected_plates.get(track_id, f"Plate {track_id}")
                self.visualizer.draw_detection(display_frame, box, text)

        return display_frame

    def process_new_plate(self, frame, box, track_id):
        # Keep track of processed plates
        self.counter.append(track_id)

        # Crop plate region
        crop = frame[box[1]:box[3], box[0]:box[2]]

        # Perform OCR
        plate_text = self.ocr_model.read_plate(crop)
        self.detected_plates[track_id] = plate_text

        if Config.SAVE_CROPS:
            crop_dir_name = os.path.join(Config.OUTPUT_DIR, "crops")
            if not os.path.exists(crop_dir_name):
                os.mkdir(crop_dir_name)
            text = self.detected_plates.get(track_id, f"Plate {track_id}")
            cv2.imwrite(os.path.join(crop_dir_name, f"plate_{text}.jpg"), crop)

    def detect_direction(self, track_id, cy, line):
        if track_id not in self.car_movements:
            self.car_movements[track_id] = []
        self.car_movements[track_id].append(cy)

        # Ensure we have enough movement history
        if len(self.car_movements[track_id]) > 2:
            prev_cy = self.car_movements[track_id][-2]
            if prev_cy < line and cy >= line:
                self.entering_count += 1 
            elif prev_cy > line and cy <= line:
                self.exiting_count += 1 

    def display_car_counts(self, frame):
        # Draw car counts on the frame
        self.visualizer.draw_car_counts(frame, self.entering_count, self.exiting_count)
        return frame