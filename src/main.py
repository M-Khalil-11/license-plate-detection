from utils.processor import PlateProcessor

def main():
    # Initialize processor
    processor = PlateProcessor()

    # Process video
    video_path = "path/to/your/video.mp4"
    detected_plates = processor.process_video(video_path)

    # Print results
    print("\nDetected License Plates:")
    for track_id, plate_text in detected_plates.items():
        print(f"Track ID: {track_id}, Plate: {plate_text}")

if __name__ == "__main__":
    main()
