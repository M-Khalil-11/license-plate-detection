# License Plate Detection and Recognition System

A system for detecting and reading license plates from video feeds using YOLO11-based models.

## Features
- Real-time license plate detection and tracking
- Optical Character Recognition (OCR) for plate reading
- Region of Interest (ROI) based processing
- Visual feedback with annotations
- Plate image cropping and saving

## Setup
1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Configure paths in `src/config/config.py`
3. Run the system:
```bash
python src/main.py
```

## Configuration
Adjust settings in `src/config/config.py`:
- Model paths
- ROI settings
- Output directory
- Display options

## Output
- Processed video display with annotations
- Cropped plate images saved to output directory
