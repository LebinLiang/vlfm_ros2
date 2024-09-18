# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import glob
import json
import os
import os.path as osp
import shutil
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from base_map import BaseMap
from geometry_utils import extract_yaw, get_rotation_matrix
from img_utils import (
    monochannel_to_inferno_rgb,
    pixel_value_within_radius,
    place_img_in_img,
    rotate_image,
)

DEBUG = False  # 调试模式标志
SAVE_VISUALIZATIONS = False  # 保存可视化标志
RECORDING = os.environ.get("RECORD_VALUE_MAP", "0") == "1"  # 记录值地图的标志
PLAYING = os.environ.get("PLAY_VALUE_MAP", "0") == "1"  # 播放记录的值地图的标志
RECORDING_DIR = "value_map_recordings"  # 记录的目录
JSON_PATH = osp.join(RECORDING_DIR, "data.json")  # JSON数据文件路径
KWARGS_JSON = osp.join(RECORDING_DIR, "kwargs.json")  # 参数JSON文件路径


class ValueMap(BaseMap):
    """生成一个地图，用于表示探索区域对找到和导航到目标对象的价值"""

    _confidence_masks: Dict[Tuple[float, float], np.ndarray] = {}
    _camera_positions: List[np.ndarray] = []
    _last_camera_yaw: float = 0.0
    _min_confidence: float = 0.25
    _decision_threshold: float = 0.35
    _map: np.ndarray

    def __init__(
        self,
        value_channels: int,
        size: int = 1000,
        use_max_confidence: bool = True,
        fusion_type: str = "default",
        obstacle_map: Optional["ObstacleMap"] = None,  # type: ignore # noqa: F821
    ) -> None:
        """
        初始化函数

        Args:
            value_channels: 值地图的通道数。
            size: 值地图的尺寸（像素）。
            use_max_confidence: 是否使用最大置信度值。
            fusion_type: 融合类型，用于与障碍地图合成。
            obstacle_map: 可选的障碍地图，用于覆盖FOV中的遮挡区域。
        """
        if PLAYING:
            size = 2000  # 如果是播放记录数据，则调整地图尺寸
        super().__init__(size)  # 初始化基类地图
        self._value_map = np.zeros((size, size, value_channels), np.float32)  # 初始化值地图
        self._value_channels = value_channels
        self._use_max_confidence = use_max_confidence
        self._fusion_type = fusion_type
        self._obstacle_map = obstacle_map
        if self._obstacle_map is not None:
            assert self._obstacle_map.pixels_per_meter == self.pixels_per_meter
            assert self._obstacle_map.size == self.size
        if os.environ.get("MAP_FUSION_TYPE", "") != "":
            self._fusion_type = os.environ["MAP_FUSION_TYPE"]

        if RECORDING:
            if osp.isdir(RECORDING_DIR):
                warnings.warn(f"Recording directory {RECORDING_DIR} already exists. Deleting it.")
                shutil.rmtree(RECORDING_DIR)
            os.mkdir(RECORDING_DIR)
            # 将所有参数转储到一个文件
            with open(KWARGS_JSON, "w") as f:
                json.dump(
                    {
                        "value_channels": value_channels,
                        "size": size,
                        "use_max_confidence": use_max_confidence,
                    },
                    f,
                )
            # 创建一个空的JSON文件
            with open(JSON_PATH, "w") as f:
                f.write("{}")

    def reset(self) -> None:
        """重置值地图到初始状态。"""
        super().reset()
        self._value_map.fill(0)

    def update_map(
        self,
        values: np.ndarray,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fov: float,
    ) -> None:
        """使用给定的深度图像、位姿和值更新值地图。

        Args:
            values: 用于更新地图的值。
            depth: 用于更新地图的深度图像，预计已归一化到[0, 1]范围。
            tf_camera_to_episodic: 从情景框架到相机框架的变换矩阵。
            min_depth: 深度值的最小值（米）。
            max_depth: 深度值的最大值（米）。
            fov: 相机的视场角（弧度）。
        """
        assert (
            len(values) == self._value_channels
        ), f"给定的值数量不正确 ({len(values)})。预期为 {self._value_channels}。"

        curr_map = self._localize_new_data(depth, tf_camera_to_episodic, min_depth, max_depth, fov)

        # 将新数据与现有数据融合
        self._fuse_new_data(curr_map, values)

        if RECORDING:
            idx = len(glob.glob(osp.join(RECORDING_DIR, "*.png")))
            img_path = osp.join(RECORDING_DIR, f"{idx:04d}.png")
            cv2.imwrite(img_path, (depth * 255).astype(np.uint8))
            with open(JSON_PATH, "r") as f:
                data = json.load(f)
            data[img_path] = {
                "values": values.tolist(),
                "tf_camera_to_episodic": tf_camera_to_episodic.tolist(),
                "min_depth": min_depth,
                "max_depth": max_depth,
                "fov": fov,
            }
            with open(JSON_PATH, "w") as f:
                json.dump(data, f)

    def sort_waypoints(
        self, waypoints: np.ndarray, radius: float, reduce_fn: Optional[Callable] = None
    ) -> Tuple[np.ndarray, List[float]]:
        """从给定的路径点中选择最佳路径点。

        Args:
            waypoints (np.ndarray): 要选择的2D路径点数组。
            radius (float): 用于选择最佳路径点的半径（米）。
            reduce_fn (Callable, optional): 用于减少给定半径内值的函数。默认为 np.max。

        Returns:
            Tuple[np.ndarray, List[float]]: 排序后的路径点及其对应的值。
        """
        radius_px = int(radius * self.pixels_per_meter)  # 将半径转换为像素

        def get_value(point: np.ndarray) -> Union[float, Tuple[float, ...]]:
            x, y = point
            px = int(-x * self.pixels_per_meter) + self._episode_pixel_origin[0]
            py = int(-y * self.pixels_per_meter) + self._episode_pixel_origin[1]
            point_px = (self._value_map.shape[0] - px, py)
            all_values = [
                pixel_value_within_radius(self._value_map[..., c], point_px, radius_px)
                for c in range(self._value_channels)
            ]
            if len(all_values) == 1:
                return all_values[0]
            return tuple(all_values)

        values = [get_value(point) for point in waypoints]

        if self._value_channels > 1:
            assert reduce_fn is not None, "当使用多个值通道时，必须提供一个减少函数。"
            values = reduce_fn(values)

        # 使用 np.argsort 获取排序后的索引
        sorted_inds = np.argsort([-v for v in values])  # type: ignore
        sorted_values = [values[i] for i in sorted_inds]
        sorted_frontiers = np.array([waypoints[i] for i in sorted_inds])

        return sorted_frontiers, sorted_values

    def visualize(
        self,
        markers: Optional[List[Tuple[np.ndarray, Dict[str, Any]]]] = None,
        reduce_fn: Callable = lambda i: np.max(i, axis=-1),
        obstacle_map: Optional["ObstacleMap"] = None,  # type: ignore # noqa: F821
    ) -> np.ndarray:
        """返回地图的图像表示"""
        # 必须取负 y 值以获得正确的方向
        reduced_map = reduce_fn(self._value_map).copy()
        if obstacle_map is not None:
            reduced_map[obstacle_map.explored_area == 0] = 0
        map_img = np.flipud(reduced_map)
        # 将值地图中的所有0值设为最大值，以免影响颜色映射（稍后会还原）
        zero_mask = map_img == 0
        map_img[zero_mask] = np.max(map_img)
        map_img = monochannel_to_inferno_rgb(map_img)
        # 将所有原本为0的值还原为白色
        map_img[zero_mask] = (255, 255, 255)

