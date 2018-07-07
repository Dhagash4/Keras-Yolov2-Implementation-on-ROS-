#! /usr/bin/env python3

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import json
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')


class getBoxes(object):

    def __init__(self,config_path,weights_path):

        with open(config_path) as config_buffer:
            self.config = json.load(config_buffer)

        ###############################
        #   Make the model
        ###############################

        yolo = YOLO(backend=self.config['model']['backend'],
                    input_size=self.config['model']['input_size'],
                    labels=self.config['model']['labels'],
                    max_box_per_image=self.config['model']['max_box_per_image'],
                    anchors=self.config['model']['anchors'])

        ###############################
        #   Load trained weights
        ###############################

        yolo.load_weights(weights_path)
        self.model = yolo
    ###############################
    #   Predict bounding boxes
    ###############################
    def predict(self,image):
        image_h, image_w, _ = image.shape
        coordinates=[]
        boxes = self.model.predict(image)
        image = draw_boxes(image, boxes, self.config['model']['labels'])

        print(len(boxes), 'boxes are found')
        for box in boxes:
            xmin = int(box.xmin*image_w)
            ymin = int(box.ymin*image_h)
            xmax = int(box.xmax*image_w)
            ymax = int(box.ymax*image_h)
            coordinates.append([xmin,ymin,xmax,ymax])
        return coordinates,image
    #cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)

       # cv2.waitKey(1)
if __name__ == '__main__':
    args = argparser.parse_args()
    print(args)
    config_path = args.conf
    weights_path = args.weights
    findBoxes = getBoxes(config_path,weights_path)

    ret =1
    cap = cv2.VideoCapture(0)
    while (ret):
        ret, image = cap.read()

        boxes,image = findBoxes.predict(image)

        cv2.imshow("output", image)
        # cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)

        cv2.waitKey(1)

