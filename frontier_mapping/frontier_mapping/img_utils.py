from typing import List, Tuple, Union
import cv2
import numpy as np

def rotate_image(
    image: np.ndarray,
    radians: float,
    border_value: Union[int, Tuple[int, int, int]] = 0,
) -> np.ndarray:
    """旋转图像到指定的角度（弧度）。

    参数:
        image (numpy.ndarray): 输入图像。
        radians (float): 旋转角度（弧度）。
        border_value (Union[int, Tuple[int, int, int]], optional): 边界的颜色值，默认为0（黑色）。

    返回:
        numpy.ndarray: 旋转后的图像。
    """
    height, width = image.shape[0], image.shape[1]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(radians), 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderValue=border_value)

    return rotated_image

def place_img_in_img(img1: np.ndarray, img2: np.ndarray, row: int, col: int) -> np.ndarray:
    """将 img2 放置到 img1 中，使得 img2 的中心位于指定的坐标（row, col）。

    参数:
        img1 (numpy.ndarray): 基础图像。
        img2 (numpy.ndarray): 要放置的图像。
        row (int): img2 中心的目标行坐标。
        col (int): img2 中心的目标列坐标。

    返回:
        numpy.ndarray: 更新后的基础图像，包含放置后的 img2。
    """
    assert 0 <= row < img1.shape[0] and 0 <= col < img1.shape[1], "像素位置超出图像范围。"
    top = row - img2.shape[0] // 2
    left = col - img2.shape[1] // 2
    bottom = top + img2.shape[0]
    right = left + img2.shape[1]

    img1_top = max(0, top)
    img1_left = max(0, left)
    img1_bottom = min(img1.shape[0], bottom)
    img1_right = min(img1.shape[1], right)

    img2_top = max(0, -top)
    img2_left = max(0, -left)
    img2_bottom = img2_top + (img1_bottom - img1_top)
    img2_right = img2_left + (img1_right - img1_left)

    img1[img1_top:img1_bottom, img1_left:img1_right] = img2[img2_top:img2_bottom, img2_left:img2_right]

    return img1

def monochannel_to_inferno_rgb(image: np.ndarray) -> np.ndarray:
    """将单通道 float32 图像转换为使用 Inferno 颜色映射的 RGB 表示。

    参数:
        image (numpy.ndarray): 输入的单通道 float32 图像。

    返回:
        numpy.ndarray: 使用 Inferno 颜色映射的 RGB 图像。
    """
    # 将输入图像归一化到[0, 1]范围
    min_val, max_val = np.min(image), np.max(image)
    peak_to_peak = max_val - min_val
    if peak_to_peak == 0:
        normalized_image = np.zeros_like(image)
    else:
        normalized_image = (image - min_val) / peak_to_peak

    # 应用 Inferno 颜色映射
    inferno_colormap = cv2.applyColorMap((normalized_image * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    return inferno_colormap

def resize_images(images: List[np.ndarray], match_dimension: str = "height", use_max: bool = True) -> List[np.ndarray]:
    """
    将图像列表中的所有图像调整为匹配其高度或宽度。

    参数:
        images (List[np.ndarray]): NumPy 图像列表。
        match_dimension (str): 指定 'height' 以匹配高度，或 'width' 以匹配宽度。
        use_max (bool): 是否使用最大尺寸（默认为 True）。

    返回:
        List[np.ndarray]: 调整后的图像列表。
    """
    if len(images) == 1:
        return images

    if match_dimension == "height":
        if use_max:
            new_height = max(img.shape[0] for img in images)
        else:
            new_height = min(img.shape[0] for img in images)
        resized_images = [
            cv2.resize(img, (int(img.shape[1] * new_height / img.shape[0]), new_height)) for img in images
        ]
    elif match_dimension == "width":
        if use_max:
            new_width = max(img.shape[1] for img in images)
        else:
            new_width = min(img.shape[1] for img in images)
        resized_images = [cv2.resize(img, (new_width, int(img.shape[0] * new_width / img.shape[1]))) for img in images]
    else:
        raise ValueError("无效的 'match_dimension' 参数。使用 'height' 或 'width'。")

    return resized_images

def crop_white_border(image: np.ndarray) -> np.ndarray:
    """裁剪图像到非白色像素的边界框。

    参数:
        image (np.ndarray): 输入图像（BGR 格式）。

    返回:
        np.ndarray: 裁剪后的图像。如果图像完全是白色，则返回原始图像。
    """
    # 将图像转换为灰度图像以便处理
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 找到非白色像素的边界框
    non_white_pixels = np.argwhere(gray_image != 255)

    if len(non_white_pixels) == 0:
        return image  # 如果图像完全是白色，则返回原始图像

    min_row, min_col = np.min(non_white_pixels, axis=0)
    max_row, max_col = np.max(non_white_pixels, axis=0)

    # 裁剪图像到边界框
    cropped_image = image[min_row : max_row + 1, min_col : max_col + 1, :]

    return cropped_image

def pad_to_square(
    img: np.ndarray,
    padding_color: Tuple[int, int, int] = (255, 255, 255),
    extra_pad: int = 0,
) -> np.ndarray:
    """
    通过在图像的左右或上下添加填充来将图像填充为正方形。

    参数:
        img (numpy.ndarray): 输入图像。
        padding_color (Tuple[int, int, int], optional): 填充颜色，默认为 (255, 255, 255)（白色）。
        extra_pad (int, optional): 额外的填充像素，默认为 0。

    返回:
        numpy.ndarray: 填充后的正方形图像。
    """
    height, width, _ = img.shape
    larger_side = max(height, width)
    square_size = larger_side + extra_pad
    padded_img = np.ones((square_size, square_size, 3), dtype=np.uint8) * np.array(padding_color, dtype=np.uint8)
    padded_img = place_img_in_img(padded_img, img, square_size // 2, square_size // 2)

    return padded_img

def pad_larger_dim(image: np.ndarray, target_dimension: int) -> np.ndarray:
    """将图像填充到指定的目标尺寸，通过添加空白边框来实现。

    参数:
        image (np.ndarray): 输入图像（高，宽，通道）。
        target_dimension (int): 目标尺寸，较大的尺寸（高度或宽度）。

    返回:
        np.ndarray: 填充后的图像（新高度，新宽度，通道）。
    """
    height, width, _ = image.shape
    larger_dimension = max(height, width)

    if larger_dimension < target_dimension:
        pad_amount = target_dimension - larger_dimension
        first_pad_amount = pad_amount // 2
        second_pad_amount = pad_amount - first_pad_amount

        if height > width:
            top_pad = np.ones((first_pad_amount, width, 3), dtype=np.uint8) * 255
            bottom_pad = np.ones((second_pad_amount, width, 3), dtype=np.uint8) * 255
            padded_image = np.vstack((top_pad, image, bottom_pad))
        else:
            left_pad = np.ones((height, first_pad_amount, 3), dtype=np.uint8) * 255
            right_pad = np.ones((height, second_pad_amount, 3), dtype=np.uint8) * 255
            padded_image = np.hstack((left_pad, image, right_pad))
    else:
        padded_image = image

    return padded_image

def pixel_value_within_radius(
    image: np.ndarray,
    pixel_location: Tuple[int, int],
    radius: int,
    reduction: str = "median",
) -> Union[float, int]:
    """返回指定像素位置半径内的最大像素值。

    参数:
        image (np.ndarray): 输入图像（二值化图像）。
        pixel_location (Tuple[int, int]): 像素位置（行，列）。
        radius (int): 半径范围。
        reduction (str, optional): 减少方法（"mean"、"max"、"median"），默认为 "median"。

    返回:
        Union[float, int]: 半径范围内的最大像素值。
    """
    # 确保像素位置在图像内
    assert (
        0 <= pixel_location[0] < image.shape[0] and 0 <= pixel_location[1] < image.shape[1]
    ), "像素位置超出图像范围。"

    top_left_x = max(0, pixel_location[0] - radius)
    top_left_y = max(0, pixel_location[1] - radius)
    bottom_right_x = min(image.shape[0], pixel_location[0] + radius + 1)
    bottom_right_y = min(image.shape[1], pixel_location[1] + radius + 1)
    cropped_image = image[top_left_x:bottom_right_x, top_left_y:bottom_right_y]

    # 绘制圆形掩码
    circle_mask = np.zeros(cropped_image.shape[:2], dtype=np.uint8)
    circle_mask = cv2.circle(
        circle_mask,
        (radius, radius),
        radius,
        color=255,
        thickness=-1,
    )
    overlap_values = cropped_image[circle_mask > 0]
    # 过滤掉值为0的像素（即尚未看到的像素）
    overlap_values = overlap_values[overlap_values > 0]
    if overlap_values.size == 0:
        return -1
    elif reduction == "mean":
        return np.mean(overlap_values)  # type: ignore
    elif reduction == "max":
        return np.max(overlap_values)
    elif reduction == "median":
        return np.median(overlap_values)  # type: ignore
    else:
        raise ValueError(f"无效的减少方法: {reduction}")

def median_blur_normalized_depth_image(depth_image: np.ndarray, ksize: int) -> np.ndarray:
    """对标准化的深度图像应用中值模糊。

    该函数首先将标准化的深度图像转换为 uint8 图像，
    然后应用中值模糊，最后将模糊后的图像转换回标准化的 float32 图像。

    参数:
        depth_image (np.ndarray): 输入的深度图像。应为标准化的 float32 图像。
        ksize (int): 中值模糊的内核大小。应为大于 1 的奇数。

    返回:
        np.ndarray: 模糊后的深度图像。为标准化的 float32 图像。
    """
    # 将标准化的深度图像转换为 uint8 图像
    depth_image_uint8 = (depth_image * 255).astype(np.uint8)

    # 应用中值模糊
    blurred_depth_image_uint8 = cv2.medianBlur(depth_image_uint8, ksize)

    # 将模糊后的图像转换回标准化的 float32 图像
    blurred_depth_image = blurred_depth_image_uint8.astype(np.float32) / 255

    return blurred_depth_image

def reorient_rescale_map(vis_map_img: np.ndarray) -> np.ndarray:
    """重新定向和缩放视觉地图图像以便显示。

    该函数对视觉地图图像进行预处理：
    1. 裁剪空白边框
    2. 将较小的尺寸填充到至少 150 像素
    3. 将图像填充为正方形
    4. 添加 50 像素的空白边框

    参数:
        vis_map_img (np.ndarray): 输入的视觉地图图像

    返回:
        np.ndarray: 重新定向和缩放后的视觉地图图像
    """
    # 去除边缘的多余白色区域
    vis_map_img = crop_white_border(vis_map_img)
    # 使图像至少为 150 像素高或宽
    vis_map_img = pad_larger_dim(vis_map_img, 150)
    # 将较短的尺寸填充到与较长的尺寸相同
    vis_map_img = pad_to_square(vis_map_img, extra_pad=50)
    # 用一些白色边框填充图像边缘
    vis_map_img = cv2.copyMakeBorder(vis_map_img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return vis_map_img

def remove_small_blobs(image: np.ndarray, min_area: int) -> np.ndarray:
    """移除图像中小于指定面积阈值的所有轮廓。

    参数:
        image (np.ndarray): 输入图像（二值图像）。
        min_area (int): 小于该阈值的轮廓将被移除。

    返回:
        np.ndarray: 移除小轮廓后的图像。
    """
    # 查找图像中的所有轮廓
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # 计算轮廓的面积
        area = cv2.contourArea(contour)

        # 如果面积小于阈值，则移除轮廓
        if area < min_area:
            cv2.drawContours(image, [contour], -1, 0, -1)

    return image

def resize_image(img: np.ndarray, new_height: int) -> np.ndarray:
    """
    将图像调整为给定的高度，同时保持纵横比。

    参数:
        img (np.ndarray): 输入图像。
        new_height (int): 目标高度。

    返回:
        np.ndarray: 调整后的图像。
    """
    # 计算纵横比
    aspect_ratio = img.shape[1] / img.shape[0]

    # 计算新的宽度
    new_width = int(new_height * aspect_ratio)

    # 调整图像大小
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_img

def fill_small_holes(depth_img: np.ndarray, area_thresh: int) -> np.ndarray:
    """
    识别深度图像中值为 0 的区域，如果区域小于给定的面积阈值，则用 1 填充。

    参数:
        depth_img (np.ndarray): 输入的深度图像
        area_thresh (int): 填充孔洞的面积阈值

    返回:
        np.ndarray: 填充小孔洞后的深度图像
    """
    # 创建一个二值图像，其中孔洞为 1，其余部分为 0
    binary_img = np.where(depth_img == 0, 1, 0).astype("uint8")

    # 查找二值图像中的轮廓
    contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    filled_holes = np.zeros_like(binary_img)

    for cnt in contours:
        # 如果轮廓的面积小于阈值
        if cv2.contourArea(cnt) < area_thresh:
            # 填充轮廓
            cv2.drawContours(filled_holes, [cnt], 0, 1, -1)

    # 创建填充后的深度图像
    filled_depth_img = np.where(filled_holes == 1, 1, depth_img)

    return filled_depth_img
