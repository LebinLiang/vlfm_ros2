import open3d as o3d
import numpy as np

# 生成一些随机点
num_points = 100
points = np.random.rand(num_points, 3)

# 创建点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# 可视化点云
o3d.visualization.draw_geometries([pcd])