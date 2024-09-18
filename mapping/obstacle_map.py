# 导入所需的模块
from typing import Any, Union

import cv2
import numpy as np
from frontier_exploration.frontier_detection import detect_frontier_waypoints
from frontier_exploration.utils.fog_of_war import reveal_fog_of_war

from base_map import BaseMap
from geometry_utils import extract_yaw, get_point_cloud, transform_points
from img_utils import fill_small_holes

class ObstacleMap(BaseMap):
    """生成两个地图: 一个表示机器人已经探索的区域，
    另一个表示机器人已经探测到的障碍物。
    """

    # 数据类型的定义
    _map_dtype: np.dtype = np.dtype(bool)  # 地图的数据类型
    _frontiers_px: np.ndarray = np.array([])  # 保存地图上的前沿点的像素位置
    frontiers: np.ndarray = np.array([])  # 保存地图上的前沿点的坐标
    radius_padding_color: tuple = (100, 100, 100)  # 表示机器人的半径填充颜色

    def __init__(
        self,
        min_height: float,  # 机器人能够探测到的最小高度
        max_height: float,  # 机器人能够探测到的最大高度
        agent_radius: float,  # 机器人的半径，用于计算导航区域
        area_thresh: float = 3.0,  # 面积阈值，单位为平方米
        hole_area_thresh: int = 100000,  # 小孔的面积阈值，单位为像素
        size: int = 1000,  # 地图的尺寸
        pixels_per_meter: int = 20,  # 每米对应的像素数
    ):
        super().__init__(size, pixels_per_meter)  # 调用父类的构造函数
        # 初始化探索区域、地图和可导航区域
        self.explored_area = np.zeros((size, size), dtype=bool)
        self._map = np.zeros((size, size), dtype=bool)
        self._navigable_map = np.zeros((size, size), dtype=bool)
        # 设置高度限制
        self._min_height = min_height
        self._max_height = max_height
        # 将面积阈值转换为像素单位
        self._area_thresh_in_pixels = area_thresh * (self.pixels_per_meter**2)
        self._hole_area_thresh = hole_area_thresh
        # 计算用于机器人半径的卷积核大小
        kernel_size = self.pixels_per_meter * agent_radius * 2
        # 将卷积核大小四舍五入为最近的奇数
        kernel_size = int(kernel_size) + (int(kernel_size) % 2 == 0)
        # 初始化用于膨胀操作的卷积核
        self._navigable_kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def reset(self) -> None:
        """重置地图，清空探索区域和障碍物地图"""
        super().reset()
        self._navigable_map.fill(0)
        self.explored_area.fill(0)
        self._frontiers_px = np.array([])
        self.frontiers = np.array([])

    def update_map(
        self,
        depth: Union[np.ndarray, Any],  # 深度图像
        tf_camera_to_episodic: np.ndarray,  # 从相机到时间坐标系的变换矩阵
        min_depth: float,  # 最小深度值
        max_depth: float,  # 最大深度值
        fx: float,  # 相机x方向上的焦距
        fy: float,  # 相机y方向上的焦距
        topdown_fov: float,  # 顶视图的视场角
        explore: bool = True,  # 是否更新探索区域
        update_obstacles: bool = True,  # 是否更新障碍物地图
    ) -> None:
        """
        根据深度图像更新障碍物地图和已探索区域。

        Args:
            depth (np.ndarray): 用于更新障碍物地图的深度图像。其值范围归一化为 [0, 1]，形状为 (高度, 宽度)。
            tf_camera_to_episodic (np.ndarray): 从相机到时间坐标系的变换矩阵。
            min_depth (float): 深度图像中的最小深度值（单位：米）。
            max_depth (float): 深度图像中的最大深度值（单位：米）。
            fx (float): 相机的 x 方向焦距。
            fy (float): 相机的 y 方向焦距。
            topdown_fov (float): 将深度相机投影到顶视图时的视场角。
            explore (bool): 是否更新已探索的区域。
            update_obstacles (bool): 是否更新障碍物地图。
        """
        if update_obstacles:
            # 如果设置了 hole_area_thresh，使用填充小孔的深度图像
            if self._hole_area_thresh == -1:
                filled_depth = depth.copy()
                filled_depth[depth == 0] = 1.0
            else:
                filled_depth = fill_small_holes(depth, self._hole_area_thresh)

            # 将深度图像缩放到实际深度范围
            scaled_depth = filled_depth * (max_depth - min_depth) + min_depth
            mask = scaled_depth < max_depth
            # 生成相机坐标系中的点云
            point_cloud_camera_frame = get_point_cloud(scaled_depth, mask, fx, fy)
            # 将点云转换到时间坐标系
            point_cloud_episodic_frame = transform_points(tf_camera_to_episodic, point_cloud_camera_frame)
            # 按照高度过滤点云，保留在指定高度范围内的点
            obstacle_cloud = filter_points_by_height(point_cloud_episodic_frame, self._min_height, self._max_height)

            # 更新顶视图中的障碍物位置
            xy_points = obstacle_cloud[:, :2]
            pixel_points = self._xy_to_px(xy_points)
            self._map[pixel_points[:, 1], pixel_points[:, 0]] = 1

            # 使用膨胀操作更新可导航区域，将障碍物地图的反向结果用于膨胀
            self._navigable_map = 1 - cv2.dilate(
                self._map.astype(np.uint8),
                self._navigable_kernel,
                iterations=1,
            ).astype(bool)

        if not explore:
            return

        # 更新已探索区域
        agent_xy_location = tf_camera_to_episodic[:2, 3]
        agent_pixel_location = self._xy_to_px(agent_xy_location.reshape(1, 2))[0]
        new_explored_area = reveal_fog_of_war(
            top_down_map=self._navigable_map.astype(np.uint8),
            current_fog_of_war_mask=np.zeros_like(self._map, dtype=np.uint8),
            current_point=agent_pixel_location[::-1],
            current_angle=-extract_yaw(tf_camera_to_episodic),
            fov=np.rad2deg(topdown_fov),
            max_line_len=max_depth * self.pixels_per_meter,
        )
        # 膨胀新探索区域
        new_explored_area = cv2.dilate(new_explored_area, np.ones((3, 3), np.uint8), iterations=1)
        self.explored_area[new_explored_area > 0] = 1
        self.explored_area[self._navigable_map == 0] = 0

        # 检查是否存在多个轮廓并选择最好的轮廓
        contours, _ = cv2.findContours(
            self.explored_area.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if len(contours) > 1:
            min_dist = np.inf
            best_idx = 0
            for idx, cnt in enumerate(contours):
                dist = cv2.pointPolygonTest(cnt, tuple([int(i) for i in agent_pixel_location]), True)
                if dist >= 0:
                    best_idx = idx
                    break
                elif abs(dist) < min_dist:
                    min_dist = abs(dist)
                    best_idx = idx
            # 更新探索区域
            new_area = np.zeros_like(self.explored_area, dtype=np.uint8)
            cv2.drawContours(new_area, contours, best_idx, 1, -1)  # type: ignore
            self.explored_area = new_area.astype(bool)

        # 计算前沿点
        self._frontiers_px = self._get_frontiers()
        if len(self._frontiers_px) == 0:
            self.frontiers = np.array([])
        else:
            self.frontiers = self._px_to_xy(self._frontiers_px)

    def _get_frontiers(self) -> np.ndarray:
        """返回地图的前沿点。"""
        # 膨胀探索区域，防止小间隙被检测为前沿点
        explored_area = cv2.dilate(
            self.explored_area.astype(np.uint8),
            np.ones((5, 5), np.uint8),
            iterations=1,
        )
        # 检测前沿点
        frontiers = detect_frontier_waypoints(
            self._navigable_map.astype(np.uint8),
            explored_area,
            self._area_thresh_in_pixels,
        )
        return frontiers

    def visualize(self) -> np.ndarray:
        """可视化地图。"""
        vis_img = np.ones((*self._map.shape[:2], 3), dtype=np.uint8) * 255
        # 以浅绿色绘制探索区域
        vis_img[self.explored_area == 1] = (200, 255, 200)
        # 以灰色绘制不可导航区域
        vis_img[self._navigable_map == 0] = self.radius_padding_color
        # 以黑色绘制障碍物
        vis_img[self._map == 1] = (0, 0, 0)
        # 以蓝色绘制前沿点
        for frontier in self._frontiers_px:
            cv2.circle(vis_img, tuple([int(i) for i in frontier]), 5, (200, 0, 0), 2)

        # 翻转图像以匹配顶部视角
        vis_img = cv2.flip(vis_img, 0)

        if len(self._camera_positions) > 0:
            self._traj_vis.draw_trajectory(
                vis_img,
                self._camera_positions,
                self._last_camera_yaw,
            )

        return vis_img

# 过滤点云中指定高度范围内的点
def filter_points_by_height(points: np.ndarray, min_height: float, max_height: float) -> np.ndarray:
    return points[(points[:, 2] >= min_height) & (points[:, 2] <= max_height)]

