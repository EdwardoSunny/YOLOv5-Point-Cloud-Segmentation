#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import open3d as o3d

from PIL import Image
from real_sense_sub import real_sense_sub


if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)

    sub = real_sense_sub()

    rospy.spin()
    # print(sub.original_image)
    # print(sub._intrinsics.fx)
    sub.crop_RGB_to_only_bounding_box()
    sub.crop_D_to_only_bounding_box()
    sub.get_cropped_image_to_point_cloud()

    img = sub.final_cropped_image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    im_pil = Image.fromarray(img)

    im_pil.show()
    print(type(sub.result_pt_cloud))
    o3d_point_cloud = o3d.geometry.PointCloud()
    o3d_point_cloud.points = o3d.utility.Vector3dVector(sub.result_pt_cloud.astype(np.float64))
    o3d.io.write_point_cloud('tennisball.ply', o3d_point_cloud)
    o3d.visualization.draw_geometries([o3d_point_cloud])


    