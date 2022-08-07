#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import open3d as o3d

from PIL import Image
from real_sense_sub import real_sense_sub
from transform import transform


if __name__ == '__main__':
    # rospy.init_node('listener', anonymous=True)

    # sub = real_sense_sub()

    # rospy.spin()
    # sub.crop_RGB_to_only_bounding_box()
    # sub.crop_D_to_only_bounding_box()
    # sub.get_cropped_image_to_point_cloud()

    # img = sub.final_cropped_image
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # im_pil = Image.fromarray(img)

    # im_pil.show()
    # print(type(sub.result_pt_cloud))
    # o3d_point_cloud = o3d.geometry.PointCloud()
    # o3d_point_cloud.points = o3d.utility.Vector3dVector(sub.result_pt_cloud.astype(np.float64))
    # o3d.io.write_point_cloud('sugar.ply', o3d_point_cloud)
    # o3d.visualization.draw_geometries([o3d_point_cloud])

    transform_info = [[-0.279217,  -0.641566,   0.714431,   0.359994],
                                          [-0.958189,   0.234405,  -0.163991, -0.0817953],
                                          [-0.0622546,  -0.730351,  -0.680221,   0.496731],
                                          [0,          0,          0,          1]]
    pcd  = o3d.io.read_point_cloud('cheezItBB.ply')
    # o3d.visualization.draw_geometries([pcd])
    trans = transform(pcd, transform_info)
    trans.remove_normals()
    # trans.show_matplot(trans.pcd_data)
    trans.cluster()
    # trans.show_matplot(trans.object)
    trans.show_o3d()
    trans.show_matplot_with_bounding_box()
    # trans.write(trans.object)

    


    