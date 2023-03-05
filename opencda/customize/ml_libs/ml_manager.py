# -*- coding: utf-8 -*-

"""
Since multiple CAV normally use the same ML/DL model,
here we have this class to enable different CAVs share the same model to
 avoid duplicate memory consumption.
"""

# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import cv2

from opencda.customize.ml_libs.detector import Detector
import numpy as np
from opencda.customize.ml_libs.ltracker import update

class MLManager(object):
    """
    A class that should contain all the ML models you want to initialize.

    Attributes
    -object_detector : torch_detector
        The YoloV5 detector load from pytorch.

    """

    def __init__(self):

        self.object_detector = Detector()



    def detect_objects(self,ims):
        yolo_detections = []
        for im in ims:
            bboxes = self.object_detector.detect(im)
            if len(bboxes) > 0:
                list_bboxs = update(bboxes, im)
                # tracker.draw_bboxes(im, list_bboxs, line_thickness=None)
                yolo_detections.append(list_bboxs)
        return yolo_detections

    def draw_2d_box(self, result, rgb_image, index):
        """
        Draw 2d bounding box based on the yolo detection.

        Args:
            -result (yolo.Result):Detection result from yolo 5.
            -rgb_image (np.ndarray): Camera rgb image.
            -index(int): Indicate the index.

        Returns:
            -rgb_image (np.ndarray): camera image with bbx drawn.
        """
        # torch.Tensor

        boundng_box = result[index]
        for detection in boundng_box:

            # the label has 80 classes, which is the same as coco dataset
            label = detection[4]
            # label_name = result.names[label]
            pos_id = detection[5]
            # if is_vehicle_cococlass(label):
            #     label_name = 'vehicle'
            # print(label_name)
            x1, y1, x2, y2 = int(
                detection[0]), int(
                detection[1]), int(
                detection[2]), int(
                detection[3])
            cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # draw text on it
            cv2.putText(rgb_image, '{}:ID-{}'.format(label, pos_id), (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 1)

        return rgb_image


def is_vehicle_cococlass(label):
    """
    Check whether the label belongs to the vehicle class according
    to coco dataset.
    Args:
        -label(int): yolo detection prediction.
    Returns:
        -is_vehicle: bool
            whether this label belongs to the vehicle class
    """
    return True if label in ['car', 'motorcycle', 'bus', 'truck'] else False;
    # vehicle_class_array = np.array([1, 2, 3, 5, 7], dtype=np.int32)
    # return True if 0 in (label - vehicle_class_array) else False
