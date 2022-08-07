import open3d as o3d

import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class transform:
    def __init__(self, pointcloud, trans_matrix):
        self.pcd_data_ply = pointcloud
        self.pcd_data_ply = self.pcd_data_ply.voxel_down_sample(voxel_size=0.005)
        self.pcd_data = np.array(self.pcd_data_ply.points) * 1/1000
        self.trans_info = np.array(trans_matrix)
        self.origin = np.array([0, 0, 0])
        self.x_unit_base = np.array([1, 0, 0])
        self.y_unit_base = np.array([0, 1, 0])
        self.z_unit_base = np.array([0, 0, 1])
        self.calc_rot()
        self.transform(self.pcd_data)
        self.pcd_data_ply.estimate_normals()
        # self.pcd_data_ply.orient_normals_to_align_with_direction([1., 0., 0.])
        self.pcd_normals_vector = self.pcd_data_ply.normals
        self.pcd_normals = np.array(self.pcd_normals_vector)
        self.transform(self.pcd_normals)
        self.x_unit_cam = np.matmul(self.rot_matrix, self.x_unit_base)
        self.y_unit_cam = np.matmul(self.rot_matrix, self.y_unit_base)
        self.z_unit_cam = np.matmul(self.rot_matrix, self.z_unit_base)
        o3d.visualization.draw_geometries([self.pcd_data_ply])

        

    def calc_rot(self):
        rot_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for i in range(0, 3):
            for j in range(0, 3):
                rot_matrix[i][j] = self.trans_info[i][j]
    
        self.translate_matrix = [self.trans_info[0, 3], self.trans_info[1, 3], self.trans_info[2, 3]]
        self.rot_matrix = np.asarray(rot_matrix)
        self.rot_matrix_inverse = inv(rot_matrix)

    def transform(self, data):
        for i in range(0, len(data)):
            data[i] = np.add(data[i], self.translate_matrix)
            data[i] = np.matmul(self.rot_matrix, data[i])

    def remove_normals(self):
        object_only = []
        normal_count = 0
        print("og data len" + str(len(self.pcd_data)))
        for i in range(0, len(self.pcd_data)):
            current_normal = self.pcd_normals[i]
            np_curr_nor = np.array(current_normal)
            np_curr_nor = np.abs(np_curr_nor)
            current_normal = np_curr_nor.tolist()
            if np.where(current_normal == np.max(current_normal))[0][0] != 2:
                object_only.append(self.pcd_data[i])
                normal_count += 1
        print("percent remove" + str(normal_count/len(self.pcd_data)))
        self.pcd_data = np.array(object_only)
        self.pcd_data_ply = o3d.geometry.PointCloud()
        self.pcd_data_ply.points = o3d.utility.Vector3dVector(self.pcd_data)
        print(self.pcd_data)
        print("cut data len" + str(len(self.pcd_data)))

    # thicc function
    def cluster(self):
        with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
        # 17.5, clusters:  2     5, clusters: 4
            labels = np.array(
                self.pcd_data_ply.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))
        # zero baesd labels

        max_label = labels.max()
        max_cluster_label = 0
        for i in range (0, max_label):
            current_cluster = self.pcd_data_ply.select_by_index(np.where(labels == i)[0].tolist())
            max_cluster = self.pcd_data_ply.select_by_index(np.where(labels == max_cluster_label)[0].tolist())
            if (len(np.array(current_cluster.points)) >= len(np.array(max_cluster.points))):
                max_cluster_label = i
        print("max cluster label: " + str(max_cluster_label))
        print("max point count:" + str(len(np.array(self.pcd_data_ply.select_by_index(np.where(labels == max_cluster_label)[0].tolist()).points))))

        # remove clusters with barely any points --> end up with 2 clusters one table one object. then easy
        # cluster_closest_idx = np.where(labels == 2)[0].tolist()
        max_cluster = np.where(labels == max_cluster_label)[0].tolist()
        print(len(max_cluster))
        # cl, ind = self.pcd_data_ply.select_by_index(max_cluster).remove_radius_outlier(nb_points=16, radius=0.4)
        cl, ind = self.pcd_data_ply.select_by_index(max_cluster).remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        self.object_ply = cl
        self.object = np.array(self.object_ply.points)

        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        self.pcd_data_ply.colors = o3d.utility.Vector3dVector(colors[:, :3])# remove clusters with barely any points --> end up with 2 clusters one table one object. then easy
        self.bounding_box = self.object_ply.get_axis_aligned_bounding_box()
        # o3d.visualization.draw_geometries([self.pcd_data_ply])
        # o3d.visualization.draw_geometries([self.object_ply])
    
    # data must of numpy arr type, NOT ply type
    def write(self, data):
        o3d_point_cloud = o3d.geometry.PointCloud()
        o3d_point_cloud.points = o3d.utility.Vector3dVector(data.astype(np.float64))
        o3d.io.write_point_cloud('cut_object.ply', o3d_point_cloud)


    def show_matplot(self, data):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.quiver(self.origin[0], self.origin[1], self.origin[2], 0.25*self.x_unit_base[0], 0.25*self.x_unit_base[1], 0.25*self.x_unit_base[2], color='r', arrow_length_ratio=0.25)
        ax.quiver(self.origin[0], self.origin[1], self.origin[2], 0.25*self.y_unit_base[0], 0.25*self.y_unit_base[1], 0.25*self.y_unit_base[2], color='g', arrow_length_ratio=0.25)
        ax.quiver(self.origin[0], self.origin[1], self.origin[2], 0.25*self.z_unit_base[0], 0.25*self.z_unit_base[1], 0.25*self.z_unit_base[2], color='b', arrow_length_ratio=0.25)


        ax.quiver(self.translate_matrix[0], self.translate_matrix[1], self.translate_matrix[2], 0.25*self.x_unit_cam[0], 0.25*self.x_unit_cam[1], 0.25*self.x_unit_cam[2], color='r', arrow_length_ratio=0.25)
        ax.quiver(self.translate_matrix[0], self.translate_matrix[1], self.translate_matrix[2], 0.25*self.y_unit_cam[0], 0.25*self.y_unit_cam[1], 0.25*self.y_unit_cam[2], color='g', arrow_length_ratio=0.25)
        ax.quiver(self.translate_matrix[0], self.translate_matrix[1], self.translate_matrix[2], 0.25*self.z_unit_cam[0], 0.25*self.z_unit_cam[1], 0.25*self.z_unit_cam[2], color='b', arrow_length_ratio=0.25)
        ax.scatter(data[0:, 0], data[0:, 1], data[0:, 2])
        plt.show()

    def show_matplot_with_bounding_box(self):
        # front bottom left, front bottom left, back bottom left, back bottom 
        bound_box_key_points = np.array(self.bounding_box.get_box_points())
        # colors = ['red', 'green', 'orange', 'yellow', 'black', 'blue', 'brown', 'purple']
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # for i in range(0, len(bound_box_key_points)):
        #     ax.scatter(bound_box_key_points[i][0], bound_box_key_points[i][1], bound_box_key_points[i][2], color=colors[i])

        center = self.pcd_data_ply.get_center()
        length = abs(bound_box_key_points[0][0] - bound_box_key_points[1][0])
        width = abs(bound_box_key_points[1][1] - bound_box_key_points[7][1])
        height = abs(bound_box_key_points[7][2] - bound_box_key_points[4][2])
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X, Y, Z = self.cuboid_data(center, (length, width, height))
        X = np.asarray(X)
        Y = np.asarray(Y)
        Z = np.asarray(Z)
        ax.plot_surface(X, Y, Z, color='b', rstride=1, cstride=1, alpha=0.1)

        ax.quiver(self.origin[0], self.origin[1], self.origin[2], 0.25*self.x_unit_base[0], 0.25*self.x_unit_base[1], 0.25*self.x_unit_base[2], color='r', arrow_length_ratio=0.25)
        ax.quiver(self.origin[0], self.origin[1], self.origin[2], 0.25*self.y_unit_base[0], 0.25*self.y_unit_base[1], 0.25*self.y_unit_base[2], color='g', arrow_length_ratio=0.25)
        ax.quiver(self.origin[0], self.origin[1], self.origin[2], 0.25*self.z_unit_base[0], 0.25*self.z_unit_base[1], 0.25*self.z_unit_base[2], color='b', arrow_length_ratio=0.25)


        ax.quiver(self.translate_matrix[0], self.translate_matrix[1], self.translate_matrix[2], 0.25*self.x_unit_cam[0], 0.25*self.x_unit_cam[1], 0.25*self.x_unit_cam[2], color='r', arrow_length_ratio=0.25)
        ax.quiver(self.translate_matrix[0], self.translate_matrix[1], self.translate_matrix[2], 0.25*self.y_unit_cam[0], 0.25*self.y_unit_cam[1], 0.25*self.y_unit_cam[2], color='g', arrow_length_ratio=0.25)
        ax.quiver(self.translate_matrix[0], self.translate_matrix[1], self.translate_matrix[2], 0.25*self.z_unit_cam[0], 0.25*self.z_unit_cam[1], 0.25*self.z_unit_cam[2], color='b', arrow_length_ratio=0.25)
        ax.scatter(self.object[0:, 0], self.object[0:, 1], self.object[0:, 2])
        plt.show()


    def show_o3d(self):
        # oriented bounding box is sus
        # obb = data.get_oriented_bounding_box()
        self.bounding_box.color = (0, 1, 0) #  green 
        o3d.visualization.draw_geometries([self.object_ply, self.bounding_box])

    def cuboid_data(self, center, size):
        # suppose axis direction: x: to left; y: to inside; z: to upper
        # get the (left, outside, bottom) point
        o = [a - b / 2 for a, b in zip(center, size)]
        # get the length, width, and height
        l, w, h = size
        x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in bottom surface
            [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in upper surface
            [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in outside surface
            [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  # x coordinate of points in inside surface
        y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in bottom surface
            [o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in upper surface
            [o[1], o[1], o[1], o[1], o[1]],          # y coordinate of points in outside surface
            [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]    # y coordinate of points in inside surface
        z = [[o[2], o[2], o[2], o[2], o[2]],                        # z coordinate of points in bottom surface
            [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],    # z coordinate of points in upper surface
            [o[2], o[2], o[2] + h, o[2] + h, o[2]],                # z coordinate of points in outside surface
            [o[2], o[2], o[2] + h, o[2] + h, o[2]]]                # z coordinate of points in inside surface
        return x, y, z