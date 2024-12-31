import os

class Config:
    # Model paths
    PLATE_MODEL_PATH = "/models/plate_model.pt"
    OCR_MODEL_PATH = "/models/ocr_model.pt"

    # ROI settings
    ROI_HEIGHT_1 = 200
    ROI_HEIGHT_2 = 300

    # Output settings
    OUTPUT_DIR = "data/output"
    SAVE_CROPS = True
    DISPLAY_RESULTS = True
    SAVE_VIDEO = True
    # OCR settings
    CONFIDENCE_THRESHOLD = 0.5

    @staticmethod
    def ensure_directories():
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
