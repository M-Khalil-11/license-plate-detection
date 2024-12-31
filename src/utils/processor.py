import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator, colors

class VideoProcessor:
    def __init__(self, input_path, output_path, detector):
        """Initialize video processor with paths and detector."""
        self.cap = cv2.VideoCapture(input_path)
        self.detector = detector
        self.setup_video(output_path)
        self.counter = []
        self.plate_texts = {}
        self.idx = 0

    def setup_video(self, output_path):
        """Setup video capture and writer."""
        assert self.cap.isOpened(), "Error reading video file"
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Define detection area
        h1, h2 = self.h-300, self.h-100
        self.area = [(0, h1), (0, h2), (self.w, h2), (self.w, h1)]
        
        # Setup video writer
        self.video_writer = cv2.VideoWriter(
            output_path, 
            cv2.VideoWriter_fourcc(*"mp4v"), 
            self.fps, 
            (self.w, self.h)
        )

    def process_frame(self, frame):
        """Process a single frame."""
        track = self.detector.plate_model.track(frame, persist=True, classes=[0])
        boxes = track[0].boxes.xyxy.int().cpu().tolist()
        track_ids = track[0].boxes.id.int().cpu().tolist()
        
        annotator = Annotator(frame, line_width=2)
        
        if boxes is not None:
            for box, track_id in zip(boxes, track_ids):
                self._process_detection(frame, box, track_id, annotator)
        
        return frame

    def _process_detection(self, frame, box, track_id, annotator):
        """Process individual detection in frame."""
        cx = int(box[0] + box[2])//2
        cy = int(box[1] + box[3])//2

        if cv2.pointPolygonTest(np.array(self.area, np.int32), ((cx, cy)), False) >= 0:
            if track_id not in self.counter:
                self.idx += 1
                self.counter.append(track_id)

                # Crop and process the license plate
                crop = frame[box[1]:box[3], box[0]:box[2]]
                ocr_results = self.detector.ocr_model(crop)
                plate_text = self.detector.process_plate_text(ocr_results)
                self.plate_texts[track_id] = plate_text
                print(f"Detected plate: {plate_text}")

        # Annotate the frame
        label = f"{self.plate_texts.get(track_id, f'Plate {track_id}')}"
        annotator.box_label(box, label, color=colors(0, True))

    def process_video(self):
        """Process the entire video."""
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            processed_frame = self.process_frame(frame)
            self.video_writer.write(processed_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cleanup()
        self.print_results()

    def cleanup(self):
        """Clean up resources."""
        self.cap.release()
        self.video_writer.release()
        cv2.destroyAllWindows()

    def print_results(self):
        """Print detection results."""
        print("\nAll detected license plates:")
        for track_id, text in self.plate_texts.items():
            print(f"Track ID: {track_id}, Plate: {text}")
