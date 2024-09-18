# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, List, Union
import cv2
import numpy as np

class TrajectoryVisualizer:
    # 初始化属性
    _num_drawn_points: int = 1  # 已绘制的点的数量
    _cached_path_mask: Union[np.ndarray, None] = None  # 缓存的路径掩膜
    _origin_in_img: Union[np.ndarray, None] = None  # 图像中的原点坐标
    _pixels_per_meter: Union[float, None] = None  # 每米的像素数
    agent_line_length: int = 10  # 代理方向线的长度
    agent_line_thickness: int = 3  # 代理方向线的粗细
    path_color: tuple = (0, 255, 0)  # 路径的颜色（RGB格式）
    path_thickness: int = 3  # 路径线的粗细
    scale_factor: float = 1.0  # 缩放因子，用于调整绘制元素的大小

    def __init__(self, origin_in_img: np.ndarray, pixels_per_meter: float):
        """
        初始化 TrajectoryVisualizer 实例。

        Args:
            origin_in_img (np.ndarray): 图像中的原点坐标。
            pixels_per_meter (float): 每米的像素数。
        """
        self._origin_in_img = origin_in_img
        self._pixels_per_meter = pixels_per_meter

    def reset(self) -> None:
        """重置已绘制的点的计数器和缓存的路径掩膜。"""
        self._num_drawn_points = 1
        self._cached_path_mask = None

    def draw_trajectory(
        self,
        img: np.ndarray,
        camera_positions: Union[np.ndarray, List[np.ndarray]],
        camera_yaw: float,
    ) -> np.ndarray:
        """
        在图像上绘制轨迹和代理，并返回更新后的图像。

        Args:
            img (np.ndarray): 输入图像。
            camera_positions (Union[np.ndarray, List[np.ndarray]]): 相机的位置列表。
            camera_yaw (float): 相机的朝向角度（弧度）。

        Returns:
            np.ndarray: 更新后的图像。
        """
        img = self._draw_path(img, camera_positions)
        img = self._draw_agent(img, camera_positions[-1], camera_yaw)
        return img

    def _draw_path(self, img: np.ndarray, camera_positions: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        绘制路径并返回更新后的图像。

        Args:
            img (np.ndarray): 输入图像。
            camera_positions (Union[np.ndarray, List[np.ndarray]]): 相机的位置列表。

        Returns:
            np.ndarray: 更新后的图像。
        """
        if len(camera_positions) < 2:
            return img
        if self._cached_path_mask is not None:
            path_mask = self._cached_path_mask.copy()
        else:
            path_mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # 绘制路径线段
        for i in range(self._num_drawn_points - 1, len(camera_positions) - 1):
            path_mask = self._draw_line(path_mask, camera_positions[i], camera_positions[i + 1])

        # 将路径颜色应用到图像
        img[path_mask == 255] = self.path_color

        self._cached_path_mask = path_mask
        self._num_drawn_points = len(camera_positions)

        return img

    def _draw_line(self, img: np.ndarray, pt_a: np.ndarray, pt_b: np.ndarray) -> np.ndarray:
        """
        绘制两点之间的线段并返回更新后的图像。

        Args:
            img (np.ndarray): 输入图像。
            pt_a (np.ndarray): 起始点坐标（公制坐标）。
            pt_b (np.ndarray): 结束点坐标（公制坐标）。

        Returns:
            np.ndarray: 更新后的图像。
        """
        # 将公制坐标转换为像素坐标
        px_a = self._metric_to_pixel(pt_a)
        px_b = self._metric_to_pixel(pt_b)

        if np.array_equal(px_a, px_b):
            return img

        # 绘制线段
        cv2.line(
            img,
            tuple(px_a[::-1]),
            tuple(px_b[::-1]),
            255,
            int(self.path_thickness * self.scale_factor),
        )

        return img

    def _draw_agent(self, img: np.ndarray, camera_position: np.ndarray, camera_yaw: float) -> np.ndarray:
        """
        绘制代理的位置和朝向，并返回更新后的图像。

        Args:
            img (np.ndarray): 输入图像。
            camera_position (np.ndarray): 代理的位置（公制坐标）。
            camera_yaw (float): 代理的朝向角度（弧度）。

        Returns:
            np.ndarray: 更新后的图像。
        """
        # 绘制代理位置
        px_position = self._metric_to_pixel(camera_position)
        cv2.circle(
            img,
            tuple(px_position[::-1]),
            int(8 * self.scale_factor),
            (255, 192, 15),
            -1,
        )
        # 绘制代理朝向
        heading_end_pt = (
            int(px_position[0] - self.agent_line_length * self.scale_factor * np.cos(camera_yaw)),
            int(px_position[1] - self.agent_line_length * self.scale_factor * np.sin(camera_yaw)),
        )
        cv2.line(
            img,
            tuple(px_position[::-1]),
            tuple(heading_end_pt[::-1]),
            (0, 0, 0),
            int(self.agent_line_thickness * self.scale_factor),
        )

        return img

    def draw_circle(self, img: np.ndarray, position: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        在图像上绘制一个圆圈（点）并返回更新后的图像。

        Args:
            img (np.ndarray): 输入图像。
            position (np.ndarray): 圆心的位置（公制坐标）。
            **kwargs: 额外的绘制参数，例如半径、颜色和厚度。

        Returns:
            np.ndarray: 更新后的图像。
        """
        px_position = self._metric_to_pixel(position)
        cv2.circle(img, tuple(px_position[::-1]), **kwargs)

        return img

    def _metric_to_pixel(self, pt: np.ndarray) -> np.ndarray:
        """
        将公制坐标转换为像素坐标。

        Args:
            pt (np.ndarray): 公制坐标。

        Returns:
            np.ndarray: 像素坐标。
        """
        # 因为图像坐标系的 y 轴方向与公制坐标系相反，需要调整 y 轴的方向
        px = pt * self._pixels_per_meter * np.array([-1, -1]) + self._origin_in_img
        px = px.astype(np.int32)
        return px

