# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Dict, Union
import cv2
import numpy as np
import open3d as o3d

# 导入几何工具函数
from geometry_utils import (
    extract_yaw,            # 提取旋转矩阵中的yaw角
    get_point_cloud,         # 从深度图像生成点云
    transform_points,        # 转换点坐标到不同的参考系
    within_fov_cone,         # 检查点云是否在相机的视野锥中
)

# 定义一个 ObjectPointCloudMap 类，用于管理物体点云地图。
class ObjectPointCloudMap:
    clouds: Dict[str, np.ndarray] = {}  # 用于存储每个物体类别的点云
    use_dbscan: bool = True             # 标志位，决定是否使用 DBSCAN 算法进行点云过滤

    def __init__(self, erosion_size: float) -> None:
        self._erosion_size = erosion_size  # 设定图像腐蚀操作的大小
        self.last_target_coord: Union[np.ndarray, None] = None  # 上一次目标物体的坐标

    def reset(self) -> None:
        """重置点云数据和目标坐标。"""
        self.clouds = {}
        self.last_target_coord = None

    def has_object(self, target_class: str) -> bool:
        """检查指定类别的物体是否存在点云数据。"""
        return target_class in self.clouds and len(self.clouds[target_class]) > 0

    def update_map(
        self,
        object_name: str,
        depth_img: np.ndarray,
        object_mask: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> None:
        """
        更新物体的点云地图。

        参数:
            object_name: 物体类别名称。
            depth_img: 深度图像。
            object_mask: 物体的二进制掩码。
            tf_camera_to_episodic: 相机到全局坐标系的转换矩阵。
            min_depth: 最小深度值。
            max_depth: 最大深度值。
            fx, fy: 相机的焦距。
        """
        # 提取当前物体的点云
        local_cloud = self._extract_object_cloud(depth_img, object_mask, min_depth, max_depth, fx, fy)
        if len(local_cloud) == 0:
            return

        # 处理过度偏移的目标物体，并对其点云标记
        if too_offset(object_mask):
            within_range = np.ones_like(local_cloud[:, 0]) * np.random.rand()
        else:
            # 对距离过远的点进行标记
            within_range = (local_cloud[:, 0] <= max_depth * 0.95) * 1.0
            within_range = within_range.astype(np.float32)
            within_range[within_range == 0] = np.random.rand()

        # 将点云从相机坐标系转换到全局坐标系
        global_cloud = transform_points(tf_camera_to_episodic, local_cloud)
        global_cloud = np.concatenate((global_cloud, within_range[:, None]), axis=1)

        # 检查物体是否太靠近机器人
        curr_position = tf_camera_to_episodic[:3, 3]
        closest_point = self._get_closest_point(global_cloud, curr_position)
        dist = np.linalg.norm(closest_point[:3] - curr_position)
        if dist < 1.0:
            return

        # 将点云更新到地图中
        if object_name in self.clouds:
            self.clouds[object_name] = np.concatenate((self.clouds[object_name], global_cloud), axis=0)
        else:
            self.clouds[object_name] = global_cloud

    def get_best_object(self, target_class: str, curr_position: np.ndarray) -> np.ndarray:
        """获取当前机器人位置与指定类别物体的最接近的目标坐标。"""
        # Check if the object is too close to the robot
        target_cloud = self.get_target_cloud(target_class)
        closest_point_2d = self._get_closest_point(target_cloud, curr_position)[:2]

        if self.last_target_coord is None:
            self.last_target_coord = closest_point_2d
        else:
            delta_dist = np.linalg.norm(closest_point_2d - self.last_target_coord)
        # Update the point cloud map
            if delta_dist < 0.1 or (delta_dist < 0.5 and np.linalg.norm(curr_position - closest_point_2d) > 2.0):
                return self.last_target_coord
            else:
                self.last_target_coord = closest_point_2d

        return self.last_target_coord

    def update_explored(self, tf_camera_to_episodic: np.ndarray, max_depth: float, cone_fov: float) -> None:
        """
        更新探索信息，移除所有被认为已经在范围内的点云，避免误报。

        参数:
            tf_camera_to_episodic: 相机到全局坐标的转换矩阵。
            max_depth: 视锥范围的最大深度。
            cone_fov: 相机的视锥角。
        """
        camera_coordinates = tf_camera_to_episodic[:3, 3]
        camera_yaw = extract_yaw(tf_camera_to_episodic)

        for obj in self.clouds:
            within_range = within_fov_cone(
                camera_coordinates, camera_yaw, cone_fov, max_depth * 0.5, self.clouds[obj]
            )
            range_ids = set(within_range[..., -1].tolist())
            for range_id in range_ids:
                if range_id == 1:
                    continue
                self.clouds[obj] = self.clouds[obj][self.clouds[obj][..., -1] != range_id]

    def get_target_cloud(self, target_class: str) -> np.ndarray:
        """获取目标类别的点云并过滤出范围内的点。"""
        target_cloud = self.clouds[target_class].copy()
        within_range_exists = np.any(target_cloud[:, -1] == 1)
        if within_range_exists:
            target_cloud = target_cloud[target_cloud[:, -1] == 1]
        return target_cloud

    def _extract_object_cloud(
        self,
        depth: np.ndarray,
        object_mask: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> np.ndarray:
        """从深度图像和物体掩码中提取物体的点云。"""
        final_mask = object_mask * 255
        final_mask = cv2.erode(final_mask, None, iterations=self._erosion_size)  # 图像腐蚀
        valid_depth = depth.copy()
        valid_depth[valid_depth == 0] = 1  # 将深度图中的洞设为最大深度
        valid_depth = valid_depth * (max_depth - min_depth) + min_depth
        cloud = get_point_cloud(valid_depth, final_mask, fx, fy)  # 从深度图和掩码提取点云
        cloud = get_random_subarray(cloud, 5000)  # 随机抽样
        if self.use_dbscan:
            cloud = open3d_dbscan_filtering(cloud)  # 使用 DBSCAN 进行点云过滤
        return cloud

    def _get_closest_point(self, cloud: np.ndarray, curr_position: np.ndarray) -> np.ndarray:
        """获取与当前机器人位置最近的点。"""
        ndim = curr_position.shape[0]
        if self.use_dbscan:
            closest_point = cloud[np.argmin(np.linalg.norm(cloud[:, :ndim] - curr_position, axis=1))]
        else:
            if ndim == 2:
                ref_point = np.concatenate((curr_position, np.array([0.5])))
            else:
                ref_point = curr_position
            distances = np.linalg.norm(cloud[:, :3] - ref_point, axis=1)
            sorted_indices = np.argsort(distances)
            top_percent = sorted_indices[: int(0.25 * len(cloud))]
            try:
                median_index = top_percent[int(len(top_percent) / 2)]
            except IndexError:
                median_index = 0
            closest_point = cloud[median_index]
        return closest_point

# Open3D 中的 DBSCAN 点云过滤器
def open3d_dbscan_filtering(points: np.ndarray, eps: float = 0.2, min_points: int = 100) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    labels = np.array(pcd.cluster_dbscan(eps, min_points))
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    non_noise_labels_mask = unique_labels != -1
    non_noise_labels = unique_labels[non_noise_labels_mask]
    non_noise_label_counts = label_counts[non_noise_labels_mask]
    if len(non_noise_labels) == 0:
        return np.array([])
    largest_cluster_label = non_noise_labels[np.argmax(non_noise_label_counts)]
    largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
    return points[largest_cluster_indices]

# 可视化和保存点云
def visualize_and_save_point_cloud(point_cloud: np.ndarray, save_path: str) -> None:
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c="b", marker="o")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig(save_path)
    plt.close()

# 从点云中随机选择指定数量的点
def get_random_subarray(points: np.ndarray, size: int) -> np.ndarray:
    if len(points) <= size:
        return points
    indices = np.random.choice(len(points), size, replace=False)
    return points[indices]

# 检查物体是否在图像的左侧或右侧偏离
def too_offset(mask: np.ndarray) -> bool:
    x, y, w, h = cv2.boundingRect(mask)
    third = mask.shape[1] // 3
    if x + w <= third:
        return x <= int(0.05 * mask.shape[1])
    elif x >= 2 * third:
        return x + w >= int(0.95 * mask.shape[1])
    else:
        return False

