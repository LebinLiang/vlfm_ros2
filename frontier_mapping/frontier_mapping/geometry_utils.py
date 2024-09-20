import math
from typing import Tuple
import numpy as np

def rho_theta(curr_pos: np.ndarray, curr_heading: float, curr_goal: np.ndarray) -> Tuple[float, float]:
    """
    计算当前位置与目标之间的极坐标（rho 和 theta），
    其中 rho 是当前位置到目标的距离，theta 是需要旋转的角度。

    Args:
        curr_pos (np.ndarray): 形状为 (2,) 的数组，表示当前的二维位置。
        curr_heading (float): 当前的朝向，以弧度表示。
        curr_goal (np.ndarray): 形状为 (2,) 的数组，表示目标的二维位置。

    Returns:
        Tuple[float, float]: 返回极坐标中的 rho 和 theta，rho 是距离，theta 是角度。
    """
    # 生成旋转矩阵，将当前的朝向转换为局部坐标系
    rotation_matrix = get_rotation_matrix(-curr_heading, ndims=2)
    
    # 计算目标相对于当前的局部位置
    local_goal = curr_goal - curr_pos
    local_goal = rotation_matrix @ local_goal

    # 计算 rho（距离）和 theta（角度）
    rho = np.linalg.norm(local_goal)
    theta = np.arctan2(local_goal[1], local_goal[0])

    return float(rho), float(theta)


def get_rotation_matrix(angle: float, ndims: int = 2) -> np.ndarray:
    """
    生成二维或三维旋转矩阵。对于二维，旋转平面为 x-y 平面；
    对于三维，绕 z 轴旋转。

    Args:
        angle (float): 旋转角度（弧度）。
        ndims (int): 旋转矩阵的维度，可以是 2 或 3。

    Returns:
        np.ndarray: 旋转矩阵。
    """
    if ndims == 2:
        return np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
    elif ndims == 3:
        return np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
    else:
        raise ValueError("ndims must be 2 or 3")


def wrap_heading(theta: float) -> float:
    """
    将给定的角度封装到 -π 和 π 之间。

    Args:
        theta (float): 角度（弧度）。
    
    Returns:
        float: 封装后的角度，范围在 -π 和 π 之间。
    """
    return (theta + np.pi) % (2 * np.pi) - np.pi


def calculate_vfov(hfov: float, width: int, height: int) -> float:
    """
    根据水平视野（HFOV）和图像传感器的宽度、高度，计算垂直视野（VFOV）。

    Args:
        hfov (float): 水平视野（弧度）。
        width (int): 图像传感器的宽度（像素）。
        height (int): 图像传感器的高度（像素）。

    Returns:
        float: 垂直视野（VFOV）以弧度表示。
    """
    # 计算对角视野（DFOV）
    dfov = 2 * math.atan(math.tan(hfov / 2) * math.sqrt((width**2 + height**2) / (width**2 + height**2)))

    # 根据对角视野计算垂直视野
    vfov = 2 * math.atan(math.tan(dfov / 2) * (height / math.sqrt(width**2 + height**2)))

    return vfov


def within_fov_cone(cone_origin: np.ndarray, cone_angle: float, cone_fov: float, cone_range: float, points: np.ndarray) -> np.ndarray:
    """
    检查一组点是否位于锥形视野范围内。

    Args:
        cone_origin (np.ndarray): 锥形视野的起点。
        cone_angle (float): 锥形视野的中心角度。
        cone_fov (float): 锥形视野的视角（弧度）。
        cone_range (float): 锥形视野的最大范围。
        points (np.ndarray): 需要检查的点集。

    Returns:
        np.ndarray: 在锥形视野范围内的点集。
    """
    # 计算每个点相对于视野起点的方向向量
    directions = points[:, :3] - cone_origin
    # 计算每个点到锥形视野起点的距离
    dists = np.linalg.norm(directions, axis=1)
    # 计算每个点与锥形中心线的角度差
    angles = np.arctan2(directions[:, 1], directions[:, 0])
    angle_diffs = np.mod(angles - cone_angle + np.pi, 2 * np.pi) - np.pi

    # 筛选出符合距离范围和角度范围的点
    mask = np.logical_and(dists <= cone_range, np.abs(angle_diffs) <= cone_fov / 2)
    return points[mask]


def convert_to_global_frame(agent_pos: np.ndarray, agent_yaw: float, local_pos: np.ndarray) -> np.ndarray:
    """
    将本地坐标系中的位置转换为全局坐标系。

    Args:
        agent_pos (np.ndarray): 代理的全局位置 (x, y, z)。
        agent_yaw (float): 代理的朝向角（弧度）。
        local_pos (np.ndarray): 本地坐标系中的位置 (x, y, z)。

    Returns:
        np.ndarray: 转换后的全局坐标。
    """
    # 将本地坐标转换为齐次坐标
    local_pos_homogeneous = np.append(local_pos, 1)

    # 生成从本地坐标系到全局坐标系的变换矩阵
    transformation_matrix = xyz_yaw_to_tf_matrix(agent_pos, agent_yaw)

    # 执行坐标变换
    global_pos_homogeneous = transformation_matrix.dot(local_pos_homogeneous)
    global_pos_homogeneous = global_pos_homogeneous[:3] / global_pos_homogeneous[-1]

    return global_pos_homogeneous


def extract_yaw(matrix: np.ndarray) -> float:
    """
    从 4x4 变换矩阵中提取 yaw 角度。

    Args:
        matrix (np.ndarray): 4x4 的变换矩阵。

    Returns:
        float: 提取出的 yaw 角度（弧度）。
    """
    # 确保输入矩阵是 4x4 的
    assert matrix.shape == (4, 4), "The input matrix must be 4x4"
    rotation_matrix = matrix[:3, :3]

    # 根据旋转矩阵计算 yaw 角
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    return yaw


def xyz_yaw_to_tf_matrix(xyz: np.ndarray, yaw: float) -> np.ndarray:
    """
    将位置 (x, y, z) 和 yaw 角转换为 4x4 变换矩阵。

    Args:
        xyz (np.ndarray): 位置向量 (x, y, z)。
        yaw (float): yaw 角度（弧度）。

    Returns:
        np.ndarray: 4x4 变换矩阵。
    """
    x, y, z = xyz
    # 构建 4x4 变换矩阵
    transformation_matrix = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0, x],
            [np.sin(yaw), np.cos(yaw), 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ]
    )
    return transformation_matrix


def closest_point_within_threshold(points_array: np.ndarray, target_point: np.ndarray, threshold: float) -> int:
    """
    在给定距离阈值内，找到距离目标点最近的点的索引。

    Args:
        points_array (np.ndarray): 2D 点集数组，格式为 (x, y)。
        target_point (np.ndarray): 目标点 (x, y)。
        threshold (float): 最大允许的距离阈值。

    Returns:
        int: 离目标点最近的点的索引，若无点满足条件，返回 -1。
    """
    # 计算每个点与目标点的距离
    distances = np.sqrt((points_array[:, 0] - target_point[0]) ** 2 + (points_array[:, 1] - target_point[1]) ** 2)
    within_threshold = distances <= threshold

    # 返回距离最近的点的索引
    if np.any(within_threshold):
        closest_index = np.argmin(distances)
        return int(closest_index)
    else:
        return -1


def transform_points(transformation_matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    对一组点应用 4x4 变换矩阵。

    Args:
        transformation_matrix (np.ndarray): 4x4 的变换矩阵。
        points (np.ndarray): 点的集合 (N, 3)。

    Returns:
        np.ndarray: 变换后的点集 (N, 3)。
    """
    if points.ndim == 1:
        points = points[:, np.newaxis]
        
    # 将点转换为齐次坐标
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])

    # 应用 4x4 变换矩阵
    transformed_points = points_homogeneous @ transformation_matrix.T

    # 转换回非齐次坐标
    return transformed_points[:, :3]


def get_point_cloud(depth_image: np.ndarray, mask: np.ndarray, fx: float, fy: float) -> np.ndarray:
    """
    根据深度图像和相机内参，生成 3D 点云。

    Args:
        depth_image (np.ndarray): 深度图像。
        mask (np.ndarray): 二值掩码图像，表示需要转换的有效像素。
        fx (float): 相机的 x 轴焦距。
        fy (float): 相机的 y 轴焦距。

    Returns:
        np.ndarray: 生成的 3D 点云，格式为 (x, y, z)。
    """
    # 计算图像中心点
    h, w = depth_image.shape
    cx, cy = w // 2, h // 2

    # 提取掩码中的有效像素坐标
    valid_pixels = np.argwhere(mask)

    # 初始化 3D 点数组
    points = []

    # 对每个有效像素，计算其 3D 坐标
    for v, u in valid_pixels:
        z = depth_image[v, u]
        if z == 0:
            continue
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points.append((x, y, z))

    return np.array(points)


def get_fov(focal_length: float, image_height_or_width: int) -> float:
    """
    计算相机的视野（FOV），根据焦距和图像尺寸。

    Args:
        focal_length (float): 焦距。
        image_height_or_width (int): 图像的高度或宽度。

    Returns:
        float: 视野（FOV）以弧度表示。
    """
    return 2 * np.arctan(image_height_or_width / (2 * focal_length))


def pt_from_rho_theta(rho: float, theta: float) -> Tuple[float, float]:
    """
    将极坐标 (rho, theta) 转换为直角坐标 (x, y)。

    Args:
        rho (float): 距离。
        theta (float): 角度（弧度）。

    Returns:
        Tuple[float, float]: 转换后的直角坐标 (x, y)。
    """
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def euler_to_matrix(roll, pitch, yaw):
    """
    将欧拉角 (roll, pitch, yaw) 转换为旋转矩阵。
    
    Args:
        roll (float): 绕x轴的旋转角（弧度）。
        pitch (float): 绕y轴的旋转角（弧度）。
        yaw (float): 绕z轴的旋转角（弧度）。
    
    Returns:
        np.ndarray: 3x3的旋转矩阵。
    """
    # 绕x轴的旋转矩阵
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # 绕y轴的旋转矩阵
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # 绕z轴的旋转矩阵
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # 组合旋转矩阵：R = Rz * Ry * Rx
    return Rz @ Ry @ Rx

def create_transformation_matrix(translation, roll, pitch, yaw):
    """
    创建从相机坐标系到baselink的4x4变换矩阵，使用欧拉角定义旋转部分。
    
    Args:
        translation (list): [tx, ty, tz]，表示平移向量。
        roll (float): 绕x轴的旋转角。
        pitch (float): 绕y轴的旋转角。
        yaw (float): 绕z轴的旋转角。
    
    Returns:
        np.ndarray: 4x4的变换矩阵。
    """
    # 生成旋转矩阵
    R = euler_to_matrix(roll, pitch, yaw)
    
    # 创建4x4的齐次变换矩阵
    T = np.eye(4)
    T[:3, :3] = R  # 设置旋转矩阵
    T[:3, 3] = translation  # 设置平移向量
    
    return T