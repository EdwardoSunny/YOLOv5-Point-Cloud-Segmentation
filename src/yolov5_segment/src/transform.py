import open3d as o3d

import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class transform:
    def __init__(self, pointcloud, trans_matrix):
        self.pcd_data = pointcloud
        self.trans_info = np.array(trans_matrix)
        self.origin = np.array([0, 0, 0])
        self.x_unit_base = np.array([1, 0, 0])
        self.y_unit_base = np.array([0, 1, 0])
        self.z_unit_base = np.array([0, 0, 1])
        self.calc_rot()
        self.transform()
        self.x_unit_cam = np.matmul(self.rot_matrix, self.x_unit_base)
        self.y_unit_cam = np.matmul(self.rot_matrix, self.y_unit_base)
        self.z_unit_cam = np.matmul(self.rot_matrix, self.z_unit_base)
        

    def calc_rot(self):
        rot_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        translate_vector = []
        for i in range(0, 3):
            for j in range(0, 3):
                rot_matrix[i][j] = self.transform_info[i][j]
    
        self.translate_matrix = [self.transform_info[0, 3], self.transform_info[1, 3], self.transform_info[2, 3]]
        self.rot_matrix = np.asarray(rot_matrix)

    def transform(self):
        for i in range(0, len(self.pcd_data)):
            self.pcd_data[i] = self.pcd_data[i] * 1/1000
            self.pcd_data[i] = np.subtract(self.pcd_data[i], self.translate_matrix)
            self.pcd_data[i] = np.matmul(self.rot_matrix_inverse, self.pcd_data[i])

    def show(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.quiver(self.origin[0], self.origin[1], self.origin[2], 0.25*self.x_unit_base[0], 0.25*self.x_unit_base[1], 0.25*self.x_unit_base[2], color='r', arrow_length_ratio=0.25)
        ax.quiver(self.origin[0], self.origin[1], self.origin[2], 0.25*self.y_unit_base[0], 0.25*self.y_unit_base[1], 0.25*self.y_unit_base[2], color='g', arrow_length_ratio=0.25)
        ax.quiver(self.origin[0], self.origin[1], self.origin[2], 0.25*self.z_unit_base[0], 0.25*self.z_unit_base[1], 0.25*self.z_unit_base[2], color='b', arrow_length_ratio=0.25)


        ax.quiver(self.translate_matrix[0], self.translate_matrix[1], self.translate_matrix[2], 0.25*self.x_unit_cam[0], 0.25*self.x_unit_cam[1], 0.25*self.x_unit_cam[2], color='r', arrow_length_ratio=0.25)
        ax.quiver(self.translate_matrix[0], self.translate_matrix[1], self.translate_matrix[2], 0.25*self.y_unit_cam[0], 0.25*self.y_unit_cam[1], 0.25*self.y_unit_cam[2], color='g', arrow_length_ratio=0.25)
        ax.quiver(self.translate_matrix[0], self.translate_matrix[1], self.translate_matrix[2], 0.25*self.z_unit_cam[0], 0.25*self.z_unit_cam[1], 0.25*self.z_unit_cam[2], color='b', arrow_length_ratio=0.25)
        ax.scatter(self.pcd_data[0:, 0], self.pcd_data[0:, 1], self.pcd_data[0:, 2])
        plt.show()