import open3d as o3d
import numpy as np

from matplotlib import cm as plt

pcd  = o3d.io.read_point_cloud('tennisball.ply')
pcd = pcd.voxel_down_sample(voxel_size=0.005)
# cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.6)
# cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=1)
# o3d.visualization.draw_geometries([cl])
# pcd = cl
o3d.visualization.draw_geometries([pcd])

with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
# 17.5, clusters:  2     5, clusters: 4
    labels = np.array(
        pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))
# zero baesd labels

max_label = labels.max()
# remove clusters with barely any points --> end up with 2 clusters one table one object. then easy
cluster_closest_idx = np.where(labels == 2)[0].tolist()
pcd_cut = pcd.select_by_index(cluster_closest_idx)

print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])# remove clusters with barely any points --> end up with 2 clusters one table one object. then easy

# o3d.visualization.draw_geometries([pcd])
o3d.visualization.draw_geometries([pcd_cut])