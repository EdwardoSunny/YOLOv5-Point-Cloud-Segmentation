import open3d as o3d
import numpy as np

from matplotlib import cm as plt

pcd = pcd = o3d.io.read_point_cloud('tennisball.ply')
# o3d.visualization.draw_geometries([pcd])

with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
# 17.5, clusters:  2     5, clusters: 4
    labels = np.array(
        pcd.cluster_dbscan(eps=5, min_points=10, print_progress=True))

test = pcd.cluster_dbscan(eps=5, min_points=10, print_progress=True)
print(type(test))

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd])