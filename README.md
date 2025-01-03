# License Plate Detection and Recognition System

A system for detecting and reading license plates from video feeds using YOLO11-based models.

## Structure Overview
```bash
license-plate-detection/
├── data/                              
│   ├── output/                        # Folder to save processed outputs
│   └── input/                         # Folder containing input videos
├── src/                               
│   ├── main.py                        # Main script
│   ├── model/                         # Folder containing OCR and license plate detection models
│   └── utils/                         
│       ├── config.py                  # Configurations
│       ├── detector.py                # License plate detection and tracking
│       ├── ocr.py                     # Text extraction (OCR)
│       ├── processor.py               # Video frame processing
│       └── visualization.py           # Drawing functions for frame annotations
├── training/                          
│   ├── License_Plate_Training         # License plate detection results
│   ├── OCR_Training                   # OCR training results
│   └── training.ipynb                 # Training notebook
├── README.md                          # Project documentation
└── requirements.txt                   # Dependency list
         
```

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

2. Configure paths in `src/utils/config.py`
3. Run the system:
```bash
python src/main.py
```

## Configuration
Adjust settings in `src/utils/config.py`:
- Model paths
- Output directory
- Display option
- Saving option
## Output
- Processed video display with annotations
- Cropped plate images saved to output directory
