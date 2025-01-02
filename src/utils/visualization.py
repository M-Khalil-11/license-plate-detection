import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator, colors # type: ignore

class Visualizer:
    @staticmethod
    def draw_roi(frame, area):
        #Draw ROI area on frame
        cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 255, 0), 2)

    @staticmethod
    def draw_detection(frame, box, text, color=(0, 255, 0)):
        #Draw detection box and text on frame
        annotator = Annotator(frame, line_width=3)
        annotator.box_label(box, text, color=colors(0, True))
    @staticmethod
    def draw_car_counts(frame, entering_count, exiting_count):
        #Draw entering and exiting car counts on the frame.
        entering_text = f"Cars In: {entering_count}"
        exiting_text = f"Cars Out: {exiting_count}"
        cv2.putText(frame, entering_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, exiting_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)