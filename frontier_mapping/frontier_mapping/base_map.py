# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, List
import numpy as np
from frontier_mapping.traj_visualizer import TrajectoryVisualizer

class BaseMap:
    # 定义类属性，用于存储摄像头位置、上一次摄像头的偏航角和地图数据类型
    _camera_positions: List[np.ndarray] = []  # 记录每个时间步的摄像头位置
    _last_camera_yaw: float = 0.0  # 记录上一次摄像头的偏航角度
    _map_dtype: np.dtype = np.dtype(np.float32)  # 地图数据类型，默认是float32

    def __init__(self, size: int = 1000, pixels_per_meter: int = 20, *args: Any, **kwargs: Any):
        """
        初始化 BaseMap 对象。
        
        参数:
            size: 地图的大小，单位是像素。
            pixels_per_meter: 每米所包含的像素数，影响地图的分辨率。
        """
        self.pixels_per_meter = pixels_per_meter  # 每米对应的像素数，用于缩放
        self.size = size  # 地图的大小，单位为像素
        # 创建一个 size x size 的空白地图，初始化为 0
        self._map = np.zeros((size, size), dtype=self._map_dtype)
        # 设置地图的原点（像素坐标系中的中心位置）
        self._episode_pixel_origin = np.array([size // 2, size // 2])
        # 初始化轨迹可视化器，用于记录和可视化机器人轨迹
        self._traj_vis = TrajectoryVisualizer(self._episode_pixel_origin, self.pixels_per_meter)

    def reset(self) -> None:
        """重置地图和轨迹可视化器，将地图清空并重新初始化。"""
        self._map.fill(0)  # 将地图上所有像素重置为 0
        self._camera_positions = []  # 清空摄像头位置记录
        # 重新初始化轨迹可视化器
        self._traj_vis = TrajectoryVisualizer(self._episode_pixel_origin, self.pixels_per_meter)

    def update_agent_traj(self, robot_xy: np.ndarray, robot_heading: float) -> None:
        """
        更新机器人轨迹。
        
        参数:
            robot_xy: 机器人在地图上的 (x, y) 坐标。
            robot_heading: 机器人的朝向（以弧度表示的偏航角）。
        """
        self._camera_positions.append(robot_xy)  # 记录当前时间步的机器人位置
        self._last_camera_yaw = robot_heading  # 更新摄像头的偏航角

    def _xy_to_px(self, points: np.ndarray) -> np.ndarray:
        """将 (x, y) 坐标转换为像素坐标。
        
        参数:
            points: 要转换的 (x, y) 坐标数组。
        
        返回:
            转换后的像素坐标数组。
        """
        # 将 (x, y) 坐标乘以 pixels_per_meter，转换为像素坐标
        # 同时将 y 轴进行反转，并加上地图的像素原点偏移
        px = np.rint(points[:, ::-1] * self.pixels_per_meter) + self._episode_pixel_origin
        # 反转 y 轴，使得像素坐标的 (0, 0) 位置在左上角
        px[:, 0] = self._map.shape[0] - px[:, 0]
        return px.astype(int)  # 返回整数像素坐标

    def _px_to_xy(self, px: np.ndarray) -> np.ndarray:
        """将像素坐标转换为 (x, y) 坐标。
        
        参数:
            px: 要转换的像素坐标数组。
        
        返回:
            转换后的 (x, y) 坐标数组。
        """
        # 复制像素坐标，防止修改原始数据
        px_copy = px.copy()
        # 反转 y 轴坐标，使像素坐标变回正常坐标
        px_copy[:, 0] = self._map.shape[0] - px_copy[:, 0]
        # 将像素坐标减去原点，并通过 pixels_per_meter 缩放为实际坐标
        points = (px_copy - self._episode_pixel_origin) / self.pixels_per_meter
        return points[:, ::-1]  # 返回 (x, y) 坐标

