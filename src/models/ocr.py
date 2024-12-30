from ultralytics import YOLO # type: ignore

class OCRModel:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self._initialize_char_map()

    def _initialize_char_map(self):
        #Initialize character mapping for OCR
        self.char_map = {}
        for k, v in self.model.names.items():
            val = int(v)
            if val < 10:  # Numbers 0-9
                self.char_map[k] = str(val)
            else:  # Letters A-Z (10-35)
                self.char_map[k] = chr(val - 10 + ord('A'))

    def read_plate(self, plate_img):
        #Perform OCR on plate image
        results = self.model(plate_img)
        if len(results[0].boxes) == 0:
            return None

        chars = []
        for box in results[0].boxes:
            x_center = (box.xyxy[0][0] + box.xyxy[0][2])/2
            class_id = int(box.cls[0])
            if class_id in self.char_map:
                chars.append((x_center.item(), self.char_map[class_id]))

        if not chars:
            return None

        # Sort characters by x position and join
        chars.sort(key=lambda x: x[0])
        return ''.join(char[1] for char in chars)
