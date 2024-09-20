#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber

from frontier_mapping.traj_visualizer import TrajectoryVisualizer

from frontier_mapping.obstacle_map import ObstacleMap
from frontier_mapping.object_point_cloud_map import ObjectPointCloudMap

from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped
import transforms3d as tfs
from frontier_mapping.geometry_utils import transform_points,create_transformation_matrix
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32

class SemanticMappingNode(Node):
    def __init__(self):
        super().__init__('semantic_mapping_node')

        self.bridge = CvBridge()
        
        #self._object_map: ObjectPointCloudMap = ObjectPointCloudMap(erosion_size=1.0)
        self._obstacle_map = ObstacleMap(
                min_height=0.4,
                max_height=0.5,
                area_thresh=1.5,
                agent_radius=0.18,
                hole_area_thresh=100000,
            )
        self.min_depth = 0.05
        self.max_depth = 8.0
        self.fx = 554
        self.fy = 554
        self.topdown_fov = 1.3

        # Initialize Subscribers for RGB image, Depth image, and Odom data
        self.rgb_sub = Subscriber(self, Image, '/rgb/image_raw')
        self.depth_sub = Subscriber(self, Image, '/rgbd/depth/image_raw')
        self.odom_sub = Subscriber(self, Odometry, '/odom')

        self.point_cloud_pub = self.create_publisher(PointCloud2, 'point_cloud_topic', 10)

        # ApproximateTimeSynchronizer for synchronizing messages with a slop of 0.1 seconds
        self.ts = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub, self.odom_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.data_callback)

        # Variables to store the current sensor data
        self.current_rgb_image = None
        self.current_depth_image = None
        self.current_odom = None

        self.robot_xy  =(0,0)
        self.robot_heading=  0.0


        # 创建一个tf2的Buffer对象，存储TF变换
        self.tf_buffer = Buffer()

        # 创建一个TransformListener对象，监听TF变换
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # trans = self.tf_buffer.lookup_transform('odom', 'rgbd_link', rclpy.time.Time())
        # self.tf_matrix = self.transform_to_matrix(trans)
        # self.get_logger().info(f'Transform Matrix:\n{transform_matrix}'

        # Create a timer that periodically processes the data
        self.timer = self.create_timer(0.1, self.timer_callback)  # Timer to call function every 0.5 seconds

    def create_point_cloud_msg(self, points):
        """
        将点云数组转换为 sensor_msgs/PointCloud2 消息格式。
        """
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "odom"  # 指定点云的坐标系，可以根据需要修改

        # 使用 sensor_msgs_py.point_cloud2 提供的工具函数将XYZ点转换为PointCloud2消息
        return create_cloud_xyz32(header, points)

    def transform_to_matrix(self, trans: TransformStamped) -> np.ndarray:
        # 从TransformStamped中获取平移和旋转（四元数）
        translation = trans.transform.translation
        rotation = trans.transform.rotation

        euler_angles = tfs.euler.quat2euler([rotation.w,rotation.x, rotation.y, rotation.z])
    

        self.get_logger().info(f"Heading (radians): {euler_angles[2]}")
        # 将四元数转换为3x3的旋转矩阵
        # rotation_matrix = tfs.quaternions.quat2mat([rotation.x, rotation.y, rotation.z, rotation.w])

        # 创建4x4的变换矩阵
        transform_matrix = np.eye(4)  # 初始化为单位矩阵
        transform_matrix = create_transformation_matrix([translation.x, translation.y, translation.z] , euler_angles[0] , euler_angles[1] ,euler_angles[2])

        # transform_matrix[:3, :3] = rotation_matrix  # 设置旋转矩阵
        # transform_matrix[:3, 3] = [translation.x, translation.y, translation.z]  # 设置平移向量

        return transform_matrix
    


    def data_callback(self, rgb_msg, depth_msg, odom_msg):
        # Convert ROS images to OpenCV format using CvBridge
        self.current_rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        self.current_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        
        self.current_odom = odom_msg
        
        position = odom_msg.pose.pose.position
        self.robot_xy = (position.x, position.y)

        # self.tf_matrix = self.create_transform_matrix(position,odom_msg.pose.pose.orientation)

        # 提取机器人的朝向（以四元数表示）
        orientation = odom_msg.pose.pose.orientation
        # 将四元数转换为欧拉角 (roll, pitch, yaw)
        euler_angles = tfs.euler.quat2euler([orientation.x, orientation.y, orientation.z, orientation.w])
        # 偏航角 yaw 就是机器人朝向

        self.robot_heading = euler_angles[2]

        #self.get_logger().info(f"Position: {self.robot_xy}")
        #self.get_logger().info(f"Heading (radians): {self.robot_heading}")
        #self.get_logger().info(f"Transform Matrix:\n{self.tf_matrix}")

    def timer_callback(self):
        # This function is called periodically by the timer
        if self.current_rgb_image is None or self.current_depth_image is None or self.current_odom is None:
            self.get_logger().warn("Waiting for all data to be available.")
            return
        
        try:
            # 查询从 'base_link' 到 'odom' 的变换
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform('odom', 'base_link', now)

            # 打印位置信息和平移向量
            #self.get_logger().info(f"Translation: {trans.transform.translation}")
            #self.get_logger().info(f"Rotation: {trans.transform.rotation}")

            self.tf_matrix = self.transform_to_matrix(trans)

        except Exception as e:
            self.get_logger().warn(f"Could not transform: {e}")


        self._obstacle_map.update_map(
                self.current_depth_image,
                self.tf_matrix,
                self.min_depth,
                self.max_depth,
                self.fx,
                self.fy,
                self.topdown_fov,
                explore=True,
            )

        point_cloud_msg = self.create_point_cloud_msg(self._obstacle_map.point_cloud_episodic_frame)
        self.point_cloud_pub.publish(point_cloud_msg)

        #self._obstacle_map.update_agent_traj(self.robot_xy, self.robot_heading)

        semantic_map = self._obstacle_map.visualize()

        self.show_map(semantic_map)

        # Process the available images and odom data
        self.get_logger().info("Processing data in timer thread...")

        # Example: Display the RGB image
        cv2.imshow("RGB Image", self.current_rgb_image)
        cv2.waitKey(1)

        # Example: Display depth image (converted to 8-bit for visualization)
        depth_display = cv2.normalize(self.current_depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imshow("Depth Image", depth_display)
        cv2.waitKey(1)

    def show_map(self, semantic_map: np.ndarray):
        # This function can be used to display or process any additional data
        cv2.imshow('Semantic Map', semantic_map)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = SemanticMappingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
