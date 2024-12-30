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
        annotator = Annotator(frame, line_width=2)
        annotator.box_label(box, text, color=colors(0, True))