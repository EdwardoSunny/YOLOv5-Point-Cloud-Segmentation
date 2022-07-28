#!/usr/bin/env python3

import rospy
import cv2
import torch
import numpy as np
import pyrealsense2 as rs

from PIL import Image as PILImage
from matplotlib import cm
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

bridge = CvBridge()

def process_once(data):
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='rgb8')
    sub_once_RGB_data.unregister()
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/edward/Documents/exp13/weights/best.pt')
    output = model(cv_image)
    output.print()
    output.show()

def listener():
    rospy.init_node('listener', anonymous=True)
    global sub_once_RGB_data
    sub_once_RGB_data = rospy.Subscriber('/camera/color/image_raw', Image, process_once)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    listener()