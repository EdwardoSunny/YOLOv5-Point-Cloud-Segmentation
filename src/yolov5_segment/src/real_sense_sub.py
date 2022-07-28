#!/usr/bin/env python3

import rospy
import cv2
import torch
import numpy as np
import pyrealsense2 as rs
import math

from PIL import Image as PILImage
from PIL import ImageDraw
from matplotlib import cm
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge


class real_sense_sub:
    def __init__(self):
        self.RGB_data_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.process_once)
        self.camera_info_data_sub = rospy.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo, self.camera_info_sub)
        self.D_data_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_info_sub)

        self.bridge = CvBridge()
        
        self.model_output = None
    
        self._intrinsics = rs.intrinsics()

        self.original_image = None
        self.labeled_image = None
        self.depth_data = None

    def crop_RGB_to_only_bounding_box(self):
        self.model_output_df = self.model_output.pandas().xyxy[0]

        self.yolo_arr = []
        for i in range(0, self.model_output_df.shape[0]):
            for j in range(0, self.model_output_df.shape[1]-3):
                self.yolo_arr.append(math.ceil(float(self.model_output_df.iat[i, j])))
        self.final_cropped_image = self.original_image[self.yolo_arr[1]:self.yolo_arr[3], self.yolo_arr[0]:self.yolo_arr[2]]

    def crop_D_to_only_bounding_box(self):
        self.depth_data = self.depth_data[self.yolo_arr[1]:self.yolo_arr[3], self.yolo_arr[0]:self.yolo_arr[2]]


    
    def process_once(self, data):
        self.original_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='rgb8')
        self.RGB_data_sub.unregister()
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/edward/Documents/weights/best.pt')
        self.model_output = model(self.original_image)
        self.model_output.print()
        self.model_output.show()

   
    def camera_info_sub(self, cameraInfo):
        self._intrinsics.width = cameraInfo.width
        self._intrinsics.height = cameraInfo.height
        self._intrinsics.ppx = cameraInfo.K[2]
        self._intrinsics.ppy = cameraInfo.K[5]
        self._intrinsics.fx = cameraInfo.K[0]
        self._intrinsics.fy = cameraInfo.K[4]
        #_intrinsics.model = cameraInfo.distortion_model
        self._intrinsics.model  = rs.distortion.none     
        self._intrinsics.coeffs = [i for i in cameraInfo.D]

    def depth_info_sub(self, depth):
        self.depth_data = self.bridge.imgmsg_to_cv2(depth, desired_encoding='passthrough')
        self.depth_data = np.array(self.depth_data, dtype=np.float32)


    def get_cropped_image_to_point_cloud(self):
        final_result_xyz = []
        for y in range(0, self.depth_data.shape[0]):
            for x in range(0, self.depth_data.shape[1]):
                result = rs.rs2_deproject_pixel_to_point(self._intrinsics, [x, y], self.depth_data[y][x])
                final_result_xyz.append(result)

        self.result_pt_cloud = np.array(final_result_xyz)
    
    


    

