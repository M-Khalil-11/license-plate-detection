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

    def process_video(self, video_path):
            cap = cv2.VideoCapture(video_path)
            assert cap.isOpened(), "Error reading video file"
            
            # Get video properties
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Initialize video writer
            if Config.SAVE_VIDEO:
                vid=video_path.split('/')[-1].split('.')[0]
                output_path = os.path.join(Config.OUTPUT_DIR, f"processed_video_{vid}.mp4")
                out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            
            # Define ROI area
            area = [(0, h-100 ), 
                    (0, h-300 ), 
                    (w, h-300 ), 
                    (w, h-100 )]
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                # Process frame
                display_frame = self.process_frame(frame, area)
                
                # Save frame if configured
                if Config.SAVE_VIDEO:
                    out.write(display_frame)
                
                # Display if configured
                if Config.DISPLAY_RESULTS:
                    cv2.imshow("License Plate Detection & OCR", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            
            cap.release()
            if Config.SAVE_VIDEO:
                out.release()
            cv2.destroyAllWindows()
            
            return self.detected_plates

    def process_frame(self, frame, area):   
        #Create a copy of the frame for visualization
        display_frame = frame.copy()
        self.visualizer.draw_roi(display_frame, area)

        # Detect and track plates
        track_results = self.plate_detector.detect_and_track(frame)
        boxes = track_results[0].boxes.xyxy.int().cpu().tolist()
        track_ids = track_results[0].boxes.id.int().cpu().tolist()

        if boxes is not None:
            for box, track_id in zip(boxes, track_ids):
                # Calculate center point of detected plate
                cx = int(box[0] + box[2])//2
                cy = int(box[1] + box[3])//2

                # Check if plate is in ROI
                in_roi = cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False) >= 0

                if in_roi:
                    if track_id not in self.counter:
                        # Process new plate
                        self.process_new_plate(frame, box, track_id)

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

