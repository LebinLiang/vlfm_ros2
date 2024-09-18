# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import List, Tuple

import numpy as np

from vlfm.vlm.blip2itm import BLIP2ITMClient


class Frontier:
    def __init__(self, xyz: np.ndarray, cosine: float):
        """
        初始化 Frontier 对象

        Args:
            xyz (np.ndarray): 前沿点的坐标。
            cosine (float): 与图像的余弦相似度编码值。
        """
        self.xyz = xyz  # 前沿点坐标
        self.cosine = cosine  # 与图像的余弦相似度编码值


class FrontierMap:
    frontiers: List[Frontier] = []  # 存储所有前沿点的列表

    def __init__(self, encoding_type: str = "cosine"):
        """
        初始化 FrontierMap 对象

        Args:
            encoding_type (str): 编码类型，默认是 'cosine'。
        """
        self.encoder: BLIP2ITMClient = BLIP2ITMClient()  # 编码器客户端

    def reset(self) -> None:
        """重置前沿点列表为空。"""
        self.frontiers = []

    def update(self, frontier_locations: List[np.ndarray], curr_image: np.ndarray, text: str) -> None:
        """
        更新前沿点列表。移除不在给定列表中的前沿点，添加新的前沿点并对其进行编码。

        Args:
            frontier_locations (List[np.ndarray]): 前沿点的坐标列表。
            curr_image (np.ndarray): 机器人当前的图像观测。
            text (str): 用于将图像与文本进行比较的文本。
        """
        # 移除不在给定列表中的前沿点。使用 np.array_equal 比较。
        self.frontiers = [
            frontier
            for frontier in self.frontiers
            if any(np.array_equal(frontier.xyz, location) for location in frontier_locations)
        ]

        # 添加任何尚未存储的前沿点。设置其余弦字段为给定图像的编码值。
        cosine = None
        for location in frontier_locations:
            if not any(np.array_equal(frontier.xyz, location) for frontier in self.frontiers):
                if cosine is None:
                    cosine = self._encode(curr_image, text)  # 对图像进行编码
                self.frontiers.append(Frontier(location, cosine))  # 添加新的前沿点

    def _encode(self, image: np.ndarray, text: str) -> float:
        """
        对给定的图像进行编码。

        Args:
            image (np.ndarray): 要编码的图像。
            text (str): 用于编码的文本。

        Returns:
            float: 图像的余弦相似度编码值。
        """
        return self.encoder.cosine(image, text)  # 使用编码器计算余弦值

    def sort_waypoints(self) -> Tuple[np.ndarray, List[float]]:
        """
        返回余弦值最高的前沿点及其余弦值。

        Returns:
            Tuple[np.ndarray, List[float]]: 排序后的前沿点坐标和对应的余弦值。
        """
        # 使用 np.argsort 获取排序后的余弦值的索引
        cosines = [f.cosine for f in self.frontiers]  # 提取所有前沿点的余弦值
        waypoints = [f.xyz for f in self.frontiers]  # 提取所有前沿点的坐标
        sorted_inds = np.argsort([-c for c in cosines])  # 按照余弦值降序排序
        sorted_values = [cosines[i] for i in sorted_inds]  # 排序后的余弦值
        sorted_frontiers = np.array([waypoints[i] for i in sorted_inds])  # 排序后的前沿点坐标

        return sorted_frontiers, sorted_values

