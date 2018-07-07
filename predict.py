#!/usr/bin/env python
import argparse
import os
import rospy 
import sys
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import json
import time
import livepredictclass

n = True

class final():
	
	def __init__(self):

		self.node_name = "yolo_image_output"
		rospy.init_node(self.node_name)
		rospy.on_shutdown(self.cleanup)		
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)
		self.image_pub = rospy.Publisher("/usb_cam/yolo_prediction",Image, queue_size=2)


	def image_callback(self,ros_image):
		
		global n
		
		try:
			frame = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
		except CvBridgeError, e:
			print e
		frame1 = np.array(frame, dtype=np.uint8)
		
		if n == True:

			abc = livepredictclass.getBoxes(config_path="/home/dhagash/catkin_ws/src/predict_opencv_pkg/keras-yolo/config.json", 
							weights_path="/home/dhagash/catkin_ws/src/predict_opencv_pkg/keras-yolo/tiny_yolo_all.h5")

	
			
			self.model = abc
			
 			n = False
			
		self.start_time = time.time()
		display_image = self.process_image(frame1)

	def process_image(self,frame):
		
		
		image = frame
		coordinates, image = self.model.predict(image)
		self.image_pub.publish(self.bridge.cv2_to_imgmsg(image,"bgr8"))
		print("FPS: ", 1.0 / (time.time() - self.start_time))
		print(coordinates, 'Coordinates')
		return image


	def cleanup(self):

		print "Shutting down vision nodes"
		cv2.destroyAllWindows()

def main(args):
	try:
		final()
		rospy.spin()

	except KeyboardInterrupt:
		print "Shutting down vision node."
		cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)
