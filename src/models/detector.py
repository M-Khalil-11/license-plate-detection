from ultralytics import YOLO  # type: ignore

class PlateDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_and_track(self, frame, persist=True):
        #Detect and track license plates in a frame
        return self.model.track(frame, persist=persist, classes=[0])