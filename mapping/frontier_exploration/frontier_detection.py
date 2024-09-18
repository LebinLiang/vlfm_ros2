import os
from typing import List, Optional

import cv2
import numpy as np
from numba import njit

from frontier_exploration.utils.bresenham_line import bresenhamline
from frontier_exploration.utils.frontier_utils import closest_line_segment

# 从环境变量中读取调试和可视化设置
VISUALIZE = os.environ.get("MAP_VISUALIZE", "False").lower() == "true"
DEBUG = os.environ.get("MAP_DEBUG", "False").lower() == "true"


def detect_frontier_waypoints(
    full_map: np.ndarray,
    explored_mask: np.ndarray,
    area_thresh: Optional[int] = -1,
    xy: Optional[np.ndarray] = None,
):
    """
    检测前沿点并计算其路径点。

    Args:
        full_map (np.ndarray): 完整的地图图像，白色表示可导航区域。
        explored_mask (np.ndarray): 已经探测过的区域的掩膜。
        area_thresh (Optional[int]): 前沿点有效的最小未探测区域（像素数）。默认值为 -1，表示不进行过滤。
        xy (Optional[np.ndarray]): 用于计算距离的参考坐标。如果为 None，则返回每个前沿的中点。

    Returns:
        np.ndarray: 每个前沿的路径点。
    """
    if DEBUG:
        import time

        os.makedirs("map_debug", exist_ok=True)
        cv2.imwrite(
            f"map_debug/{int(time.time())}_debug_full_map_{area_thresh}.png", full_map
        )
        cv2.imwrite(
            f"map_debug/{int(time.time())}_debug_explored_mask_{area_thresh}.png",
            explored_mask,
        )

    if VISUALIZE:
        img = cv2.cvtColor(full_map * 255, cv2.COLOR_GRAY2BGR)
        img[explored_mask > 0] = (127, 127, 127)

        cv2.imshow("inputs", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    explored_mask[full_map == 0] = 0
    frontiers = detect_frontiers(full_map, explored_mask, area_thresh)
    if VISUALIZE:
        img = cv2.cvtColor(full_map * 255, cv2.COLOR_GRAY2BGR)
        img[explored_mask > 0] = (127, 127, 127)
        # 绘制每个前沿点的线段
        for idx, frontier in enumerate(frontiers):
            # 从 COLORMAP_RAINBOW 中均匀采样颜色
            color = cv2.applyColorMap(
                np.uint8([255 * (idx + 1) / len(frontiers)]), cv2.COLORMAP_RAINBOW
            )[0][0]
            color = tuple(int(i) for i in color)
            for idx2, p in enumerate(frontier):
                if idx2 < len(frontier) - 1:
                    cv2.line(img, p[0], frontier[idx2 + 1][0], color, 3)
        cv2.imshow("frontiers", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    waypoints = frontier_waypoints(frontiers, xy)
    return waypoints


def detect_frontiers(
    full_map: np.ndarray, explored_mask: np.ndarray, area_thresh: Optional[int] = -1
) -> List[np.ndarray]:
    """
    在地图中检测前沿点。

    Args:
        full_map (np.ndarray): 白色多边形在黑色图像上，其中白色表示可导航区域。
        explored_mask (np.ndarray): 已经探测过的区域的掩膜。
        area_thresh (Optional[int]): 前沿点有效的最小未探测区域（像素数）。默认值为 -1，表示不进行过滤。

    Returns:
        List[np.ndarray]: 前沿点的列表。
    """
    # 过滤掉小的未探测区域
    filtered_explored_mask = filter_out_small_unexplored(
        full_map, explored_mask, area_thresh
    )
    contours, _ = cv2.findContours(
        filtered_explored_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    if VISUALIZE:
        img = cv2.cvtColor(full_map * 255, cv2.COLOR_GRAY2BGR)
        img[explored_mask > 0] = (127, 127, 127)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
        cv2.imshow("contours", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    unexplored_mask = np.where(filtered_explored_mask > 0, 0, full_map)
    unexplored_mask = cv2.blur(  # 模糊处理以增加一些宽容度
        np.where(unexplored_mask > 0, 255, unexplored_mask), (3, 3)
    )
    frontiers = []
    # TODO: 应该只有一个轮廓（地图上只有一个探测区域）
    for contour in contours:
        frontiers.extend(
            contour_to_frontiers(interpolate_contour(contour), unexplored_mask)
        )
    return frontiers


def filter_out_small_unexplored(
    full_map: np.ndarray, explored_mask: np.ndarray, area_thresh: int
):
    """
    过滤掉小的未探测区域，这些区域在前沿检测中将被忽略。

    Args:
        full_map (np.ndarray): 完整的地图图像。
        explored_mask (np.ndarray): 已经探测过的区域的掩膜。
        area_thresh (int): 最小未探测区域（像素数）阈值。

    Returns:
        np.ndarray: 过滤后的掩膜。
    """
    if area_thresh == -1:
        return explored_mask

    unexplored_mask = full_map.copy()
    unexplored_mask[explored_mask > 0] = 0

    if VISUALIZE:
        img = cv2.cvtColor(unexplored_mask * 255, cv2.COLOR_GRAY2BGR)
        cv2.imshow("unexplored mask", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 找到未探测掩膜中的轮廓
    contours, _ = cv2.findContours(
        unexplored_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if VISUALIZE:
        img = cv2.cvtColor(unexplored_mask * 255, cv2.COLOR_GRAY2BGR)
        # 以红色绘制轮廓
        cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
        cv2.imshow("unexplored mask with contours", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 将小的未探测区域添加到已探测掩膜中
    small_contours = []
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < area_thresh:
            mask = np.zeros_like(explored_mask)
            mask = cv2.drawContours(mask, [contour], 0, 1, -1)
            masked_values = unexplored_mask[mask.astype(bool)]
            values = set(masked_values.tolist())
            if 1 in values and len(values) == 1:
                small_contours.append(contour)
    new_explored_mask = explored_mask.copy()
    cv2.drawContours(new_explored_mask, small_contours, -1, 255, -1)

    if VISUALIZE and len(small_contours) > 0:
        # 绘制完整地图和新的已探测掩膜，然后勾勒出添加到已探测掩膜中的轮廓
        img = cv2.cvtColor(full_map * 255, cv2.COLOR_GRAY2BGR)
        img[new_explored_mask > 0] = (127, 127, 127)
        cv2.drawContours(img, small_contours, -1, (0, 0, 255), 3)
        cv2.imshow("small unexplored areas", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return new_explored_mask


def interpolate_contour(contour):
    """
    给定一个 cv2 轮廓，使用 Bresenham 算法在每对点之间添加点，使轮廓更加连续。

    Args:
        contour: cv2 轮廓，形状为 (N, 1, 2)

    Returns:
        np.ndarray: 连续的轮廓点。
    """
    # 首先，将前沿点重新调整为形状为 (N-1, 2, 2) 的 2D 数组，表示相邻点之间的线段
    line_segments = np.concatenate((contour[:-1], contour[1:]), axis=1).reshape(
        (-1, 2, 2)
    )
    # 还添加一个将最后一点连接到第一点的线段
    line_segments = np.concatenate(
        (line_segments, np.array([contour[-1], contour[0]]).reshape((1, 2, 2)))
    )
    pts = []
    for (x0, y0), (x1, y1) in line_segments:
        pts.append(
            bresenhamline(np.array([[x0, y0]]), np.array([[x1, y1]]), max_iter=-1)
        )
    pts = np.concatenate(pts).reshape((-1, 1, 2))
    return pts


@njit
def contour_to_frontiers(contour, unexplored_mask):
    """
    将 OpenCV 的轮廓转换为前沿点列表，每个列表包含一个连续的前沿点集合。

    Args:
        contour: OpenCV 轮廓。
        unexplored_mask: 未探测区域的掩膜。

    Returns:
        List[np.ndarray]: 每个前沿的点数组。
    """
    bad_inds = []
    num_contour_points = len(contour)
    for idx in range(num_contour_points):
        x, y = contour[idx][0]
        if unexplored_mask[y, x] == 0:
            bad_inds.append(idx)
    frontiers = np.split(contour, bad_inds)
    # np.split 很快但不会删除分割索引处的元素
    filtered_frontiers = []
    front_last_split = (
        0 not in bad_inds
        and len(bad_inds) > 0
        and max(bad_inds) < num_contour_points - 2
    )
    for idx, f in enumerate(frontiers):
        # 一个前沿必须至少有 2 个点（3 个点与坏索引）
        if len(f) > 2 or (idx == 0 and front_last_split):
            if idx == 0:
                filtered_frontiers.append(f)
            else:
                filtered_frontiers.append(f[1:])
    # 如果第一个前沿的第一个点和最后一个前沿的最后一个点是原始轮廓的第一个和最后一个点，则将第一个和最后一个前沿合并
    if len(filtered_frontiers) > 1 and front_last_split:
        last_frontier = filtered_frontiers.pop()
        filtered_frontiers[0] = np.concatenate((last_frontier, filtered_frontiers[0]))
    return filtered_frontiers


def frontier_waypoints(
    frontiers: List[np.ndarray], xy: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    对于每个前沿，返回离给定坐标最近的点。如果没有给定坐标，则返回每个前沿的中点。

    Args:
        frontiers (List[np.ndarray]): 每个数组都是一个前沿点的数组。
        xy (Optional[np.ndarray]): 给定坐标。

    Returns:
        np.ndarray: 每个前沿的路径点。
    """
    if xy is None:
        return np.array([get_frontier_midpoint(i) for i in frontiers])
    return np.array([get_closest_frontier_point(xy, i) for i in frontiers])


@njit
def get_frontier_midpoint(frontier) -> np.ndarray:
    """
    计算前沿的中点。

    Args:
        frontier: 前沿的点数组。

    Returns:
        np.ndarray: 前沿的中点。
    """
    # 首先，将前沿点重新调整为形状为 (X, 2, 2) 的 2D 数组，表示相邻点之间的线段
    line_segments = np.concatenate((frontier[:-1], frontier[1:]), axis=1).reshape(
        (-1, 2, 2)
    )
    # 计算每个线段的长度
    line_lengths = np.sqrt(
        np.square(line_segments[:, 0, 0] - line_segments[:, 1, 0])
        + np.square(line_segments[:, 0, 1] - line_segments[:, 1, 1])
    )
    cum_sum = np.cumsum(line_lengths)
    total_length = cum_sum[-1]
    # 找到前沿的中点
    half_length = total_length / 2
    # 找到包含中点的线段
    line_segment_idx = np.argmax(cum_sum > half_length)
    # 计算中点坐标
    line_segment = line_segments[line_segment_idx]
    line_length = line_lengths[line_segment_idx]
    # 使用中点长度与累计长度之间的差异来找到中点在该线段的位置比例
    length_up_to = cum_sum[line_segment_idx - 1] if line_segment_idx > 0 else 0
    proportion = (half_length - length_up_to) / line_length
    # 计算中点坐标
    midpoint = line_segment[0] + proportion * (line_segment[1] - line_segment[0])
    return midpoint


def get_closest_frontier_point(xy, frontier):
    """
    返回前沿中离给定坐标最近的点。

    Args:
        xy: 给定坐标。
        frontier: 前沿的点数组。

    Returns:
        np.ndarray: 离给定坐标最近的前沿点。
    """
    # 首先，将前沿点重新调整为形状为 (X, 2) 的 2D 数组，表示相邻点之间的线段
    line_segments = np.concatenate([frontier[:-1], frontier[1:]], axis=1).reshape(
        (-1, 2, 2)
    )
    closest_segment, closest_point = closest_line_segment(xy, line_segments)
    return closest_point


if __name__ == "__main__":
    import argparse
    import time

    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--full_map", help="完整地图图像路径", default="full_map.png"
    )
    parser.add_argument(
        "-e",
        "--explored_mask",
        help="已探测掩膜图像路径",
        default="explored_mask.png",
    )
    parser.add_argument(
        "-a",
        "--area_thresh",
        help="前沿点有效的最小未探测区域（像素数）",
        type=float,
        default=-1,
    )
    parser.add_argument(
        "-n",
        "--num-iterations",
        help="运行算法的次数（用于计时）。设置为 0 表示不计时",
        type=int,
        default=500,
    )
    args = parser.parse_args()

    if VISUALIZE:
        args.num_iterations = 0

    # 读取地图图像
    full_map = cv2.imread(args.full_map, 0)
    # 读取已探测掩膜图像
    explored_mask = cv2.imread(args.explored_mask, 0)
    times = []
    for _ in range(args.num_iterations + 1):
        start_time = time.time()
        waypoints = detect_frontier_waypoints(full_map, explored_mask, args.area_thresh)
        times.append(time.time() - start_time)
    if args.num_iterations > 0:
        # 跳过第一次运行，因为它由于 JIT 编译而较慢
        print(
            f"算法平均运行时间（{args.num_iterations} 次运行）：",
            np.mean(times[1:]),
        )

    # 绘制结果
    plt.figure(figsize=(10, 10))
    plt.imshow(full_map, cmap="gray")
    plt.imshow(explored_mask, cmap="gray", alpha=0.5)
    for waypoint in waypoints:
        plt.scatter(waypoint[0], waypoint[1], c="red", s=50)
    plt.show()

