#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file enhanced_simplified_camera_subscriber_node.py
@author Guo1ZY
@brief Enhanced ROS node for apple detection with improved depth processing
@version 0.2
@date 2025-06-28
"""

import rospy
import cv2
import numpy as np
import os
from threading import Lock

# ROS imports
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError

# Custom message for detection results
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String, Float32MultiArray

# YOLOv8 imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    rospy.logwarn("ultralytics not installed. Install with: pip install ultralytics")
    YOLO_AVAILABLE = False
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
class AppleDetectionNode:
    def __init__(self):
        """初始化苹果检测节点"""
        
        # 初始化ROS节点
        rospy.init_node('apple_detection_node', anonymous=True)
        
        # 参数设置
        self.setup_parameters()
        
        # 初始化变量
        self.bridge = CvBridge()
        self.data_lock = Lock()
        
        # 图像数据
        self.current_color_image = None
        self.current_depth_image = None
        self.color_camera_info = None
        
        # 相机内参矩阵
        self.color_K = None
        
        # 数据接收状态
        self.color_received = False
        self.depth_received = False
        self.camera_info_received = False
        
        # YOLOv8 检测器
        self.yolo_model = None
        self.yolo_initialized = False
        
        # 深度处理参数
        self.depth_filter_radius = 3  # 中心点周围正负3个像素
        self.depth_filter_size = 2 * self.depth_filter_radius + 1  # 7x7窗口
        
        # 初始化YOLO
        self.initialize_yolo()
        
        # 设置ROS订阅器和发布器
        self.setup_publishers()
        self.setup_subscribers()
        
        rospy.loginfo("Enhanced apple detection node initialized!")
        rospy.loginfo("Waiting for camera data...")

    def setup_parameters(self):
        """设置ROS参数"""
        # 模型路径
        self.model_path = rospy.get_param('~model_path', 
            '/home/zy/ws_livox/src/apple_detection/yolo/yolov8/model/best.pt')
        
        # 检测参数
        self.conf_threshold = rospy.get_param('~conf_threshold', 0.8)
        self.iou_threshold = rospy.get_param('~iou_threshold', 0.6)
        self.yolo_input_size = rospy.get_param('~yolo_input_size', 256)  # YOLO输入尺寸：256x256
        
        # 图像尺寸
        self.image_width = rospy.get_param('~image_width', 640)
        self.image_height = rospy.get_param('~image_height', 400)
        
        # 深度有效范围
        self.min_depth = rospy.get_param('~min_depth', 0.1)  # 米
        self.max_depth = rospy.get_param('~max_depth', 0.8)  # 米
        
        # 深度滤波参数
        self.depth_filter_radius = rospy.get_param('~depth_filter_radius', 3)  # 滤波半径
        self.min_valid_depth_points = rospy.get_param('~min_valid_depth_points', 5)  # 最少有效深度点数
        
        # 话题名称
        self.color_topic = rospy.get_param('~color_topic', '/camera/color/image_raw')
        self.depth_topic = rospy.get_param('~depth_topic', '/camera/depth/image_raw')
        self.color_info_topic = rospy.get_param('~color_info_topic', '/camera/color/camera_info')
        
        # 发布话题名称
        self.detection_topic = rospy.get_param('~detection_topic', '/apple_detection/coordinates')
        self.detection_image_topic = rospy.get_param('~detection_image_topic', '/apple_detection/image')
        
        # 显示设置
        self.enable_display = rospy.get_param('~enable_display', True)

    def setup_publishers(self):
        """设置ROS发布器"""
        # 发布检测到的苹果3D坐标
        self.coordinate_pub = rospy.Publisher(
            self.detection_topic, 
            Float32MultiArray, 
            queue_size=10
        )
        
        # 发布带检测框的图像（可选）
        self.detection_image_pub = rospy.Publisher(
            self.detection_image_topic, 
            Image, 
            queue_size=1
        )
        
        rospy.loginfo(f"Publishing coordinates to: {self.detection_topic}")
        rospy.loginfo(f"Publishing detection images to: {self.detection_image_topic}")

    def setup_subscribers(self):
        """设置ROS订阅器"""
        try:
            # 彩色图像订阅器
            self.color_sub = rospy.Subscriber(
                self.color_topic, Image, self.color_image_callback, queue_size=1)
            
            # 深度图像订阅器  
            self.depth_sub = rospy.Subscriber(
                self.depth_topic, Image, self.depth_image_callback, queue_size=1)
            
            # 相机信息订阅器（只需要彩色相机的）
            self.color_info_sub = rospy.Subscriber(
                self.color_info_topic, CameraInfo, self.color_info_callback, queue_size=1)
                
            rospy.loginfo("Subscribers setup completed")
                
        except Exception as e:
            rospy.logerr(f"Failed to setup subscribers: {str(e)}")

    def initialize_yolo(self):
        """初始化YOLOv8检测器"""
        if not YOLO_AVAILABLE:
            rospy.logwarn("YOLOv8 not available, skipping initialization")
            return
            
        try:
            # 检查模型文件是否存在
            if not os.path.exists(self.model_path):
                rospy.logerr(f"Model file not found: {self.model_path}")
                return
                
            rospy.loginfo("Initializing YOLOv8 detector...")
            
            # 加载模型
            self.yolo_model = YOLO(self.model_path)
            
            # 设置模型参数
            self.yolo_model.conf = self.conf_threshold
            self.yolo_model.iou = self.iou_threshold
            
            self.yolo_initialized = True
            rospy.loginfo("YOLOv8 detector initialized successfully!")
            
        except Exception as e:
            rospy.logerr(f"Failed to initialize YOLOv8: {str(e)}")
            self.yolo_initialized = False

    def color_image_callback(self, msg):
        """彩色图像回调函数"""
        try:
            # 转换ROS图像消息为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 检查图像尺寸是否符合预期
            if cv_image.shape[1] != self.image_width or cv_image.shape[0] != self.image_height:
                # 调整图像尺寸
                cv_image = cv2.resize(cv_image, (self.image_width, self.image_height))
            
            with self.data_lock:
                self.current_color_image = cv_image.copy()
                self.color_received = True
            
            # 如果深度图像和相机信息都已接收，进行检测
            if self.depth_received and self.camera_info_received:
                self.process_detection()
            
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge error in color callback: {str(e)}")
        except Exception as e:
            rospy.logerr(f"Error in color image callback: {str(e)}")

    def depth_image_callback(self, msg):
        """深度图像回调函数"""
        try:
            # 根据编码格式转换深度图像
            if msg.encoding == "16UC1":
                cv_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            elif msg.encoding == "32FC1":
                cv_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, msg.encoding)
            
            # 调整深度图像尺寸以匹配彩色图像
            if cv_image.shape[1] != self.image_width or cv_image.shape[0] != self.image_height:
                cv_image = cv2.resize(cv_image, (self.image_width, self.image_height), 
                                    interpolation=cv2.INTER_NEAREST)
            
            with self.data_lock:
                self.current_depth_image = cv_image.copy()
                self.depth_received = True
            
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge error in depth callback: {str(e)}")
        except Exception as e:
            rospy.logerr(f"Error in depth image callback: {str(e)}")

    def color_info_callback(self, msg):
        """彩色相机信息回调函数"""
        try:
            with self.data_lock:
                self.color_camera_info = msg
                # 提取内参矩阵
                self.color_K = np.array(msg.K).reshape(3, 3)
                self.camera_info_received = True
            
            rospy.loginfo_once(f"Received color camera info: "
                              f"fx={msg.K[0]:.2f}, fy={msg.K[4]:.2f}, "
                              f"cx={msg.K[2]:.2f}, cy={msg.K[5]:.2f}")
                              
        except Exception as e:
            rospy.logerr(f"Error in color info callback: {str(e)}")

    def detect_objects(self, image):
        """使用YOLOv8检测目标"""
        try:
            if not self.yolo_initialized or self.yolo_model is None:
                return []
            
            # 运行检测，指定输入尺寸为256x256
            results = self.yolo_model(image, imgsz=self.yolo_input_size, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 提取检测信息
                        xyxy = box.xyxy[0].cpu().numpy()  # 边界框坐标
                        conf = float(box.conf[0])  # 置信度
                        cls = int(box.cls[0])  # 类别ID
                        
                        # 获取类别名称
                        class_name = self.yolo_model.names[cls] if cls < len(self.yolo_model.names) else f"class_{cls}"
                        
                        # 只保留apple类别的检测结果
                        if class_name.lower() == 'apple':
                            # 计算中心点坐标
                            center_x = int((xyxy[0] + xyxy[2]) / 2)
                            center_y = int((xyxy[1] + xyxy[3]) / 2)
                            
                            detection = {
                                'bbox': xyxy,  # [x1, y1, x2, y2]
                                'center': (center_x, center_y),  # 中心点坐标
                                'confidence': conf,
                                'class_id': cls,
                                'class_name': class_name
                            }
                            detections.append(detection)
            
            return detections
            
        except Exception as e:
            rospy.logerr(f"Error in object detection: {str(e)}")
            return []

    def create_gaussian_weights(self, radius):
        """创建高斯权重矩阵"""
        try:
            size = 2 * radius + 1
            weights = np.zeros((size, size), dtype=np.float32)
            center = radius
            sigma = radius / 3.0  # 标准差设为半径的1/3
            
            for i in range(size):
                for j in range(size):
                    distance_sq = (i - center) ** 2 + (j - center) ** 2
                    weights[i, j] = np.exp(-distance_sq / (2 * sigma ** 2))
            
            # 归一化权重
            weights = weights / np.sum(weights)
            
            return weights
            
        except Exception as e:
            rospy.logerr(f"Error creating gaussian weights: {str(e)}")
            return np.ones((2 * radius + 1, 2 * radius + 1)) / ((2 * radius + 1) ** 2)

    def get_filtered_depth_value(self, u, v):
        """获取指定像素周围区域的加权滤波深度值"""
        try:
            if (self.current_depth_image is None or
                u < 0 or u >= self.current_depth_image.shape[1] or
                v < 0 or v >= self.current_depth_image.shape[0]):
                return None
            
            # 计算采样区域
            radius = self.depth_filter_radius
            u_start = max(0, u - radius)
            u_end = min(self.current_depth_image.shape[1], u + radius + 1)
            v_start = max(0, v - radius)
            v_end = min(self.current_depth_image.shape[0], v + radius + 1)
            
            # 获取深度值区域
            depth_region = self.current_depth_image[v_start:v_end, u_start:u_end]
            
            # 转换深度值
            if self.current_depth_image.dtype == np.uint16:
                depth_values = depth_region.astype(np.float32) / 1000.0  # 毫米转米
            elif self.current_depth_image.dtype == np.float32:
                depth_values = depth_region.copy()
            else:
                return None
            
            # 筛选有效深度值
            valid_mask = (depth_values > self.min_depth) & (depth_values < self.max_depth)
            
            if np.sum(valid_mask) < self.min_valid_depth_points:
                rospy.logdebug(f"Not enough valid depth points: {np.sum(valid_mask)}")
                return None
            
            # 创建对应的权重矩阵
            actual_height, actual_width = depth_region.shape
            
            # 创建高斯权重（根据实际区域大小）
            if actual_height == 2 * radius + 1 and actual_width == 2 * radius + 1:
                # 完整的7x7区域
                weights = self.create_gaussian_weights(radius)
            else:
                # 边界区域，创建对应大小的权重
                temp_radius = min(actual_height // 2, actual_width // 2)
                weights = self.create_gaussian_weights(temp_radius)
                weights = weights[:actual_height, :actual_width]
            
            # 只对有效深度值进行加权平均
            valid_depths = depth_values[valid_mask]
            valid_weights = weights[valid_mask]
            
            # 归一化权重
            valid_weights = valid_weights / np.sum(valid_weights)
            
            # 计算加权平均深度
            filtered_depth = np.sum(valid_depths * valid_weights)
            
            rospy.logdebug(f"Filtered depth at ({u}, {v}): {filtered_depth:.3f}m "
                          f"from {np.sum(valid_mask)} valid points")
            
            return filtered_depth
            
        except Exception as e:
            rospy.logerr(f"Error in depth filtering: {str(e)}")
            return None

    def get_3d_coordinate(self, u, v):
        """获取指定像素的3D坐标（使用滤波后的深度值）"""
        try:
            if (self.current_depth_image is None or 
                self.color_K is None or
                u < 0 or u >= self.current_depth_image.shape[1] or
                v < 0 or v >= self.current_depth_image.shape[0]):
                return None
            
            # 获取滤波后的深度值
            depth_value = self.get_filtered_depth_value(u, v)
            
            if depth_value is None:
                return None
            
            # 相机内参
            fx = self.color_K[0, 0]
            fy = self.color_K[1, 1]
            cx = self.color_K[0, 2]
            cy = self.color_K[1, 2]
            
            # 像素坐标转3D坐标（相机坐标系）
            x = (u - cx) * depth_value / fx
            y = (v - cy) * depth_value / fy
            z = depth_value
            
            return (x, y, z)
            
        except Exception as e:
            rospy.logerr(f"Error getting 3D coordinate: {str(e)}")
            return None

    def process_detection(self):
        """主要检测处理函数"""
        try:
            if (self.current_color_image is None or 
                self.current_depth_image is None or 
                not self.yolo_initialized):
                return
            
            # 进行YOLO检测（已经在detect_objects中过滤了apple类别）
            detections = self.detect_objects(self.current_color_image)
            
            if not detections:
                rospy.logdebug("No apple detections found")
                return
            
            # 创建用于显示的图像
            display_image = self.current_color_image.copy()
            
            # 处理每个apple检测结果
            valid_apples = []
            
            for i, detection in enumerate(detections):
                center_x, center_y = detection['center']
                confidence = detection['confidence']
                class_name = detection['class_name']
                
                # 获取中心点的3D坐标（使用滤波）
                coord_3d = self.get_3d_coordinate(center_x, center_y)
                
                if coord_3d is not None:
                    x, y, z = coord_3d
                    
                    # 添加到有效苹果列表
                    apple_data = {
                        'class_name': class_name,
                        'confidence': confidence,
                        'center_2d': (center_x, center_y),
                        'center_3d': (x, y, z),
                        'detection_id': i,
                        'distance': z  # 用于排序
                    }
                    valid_apples.append(apple_data)
                    
                    # 在图像上绘制检测结果
                    self.draw_detection_result(display_image, detection, coord_3d)
                    
                    # 打印检测信息
                    rospy.loginfo_throttle(1.0, 
                        f"Detected {class_name} at 3D: ({x:.3f}, {y:.3f}, {z:.3f}), "
                        f"2D: ({center_x}, {center_y}), confidence: {confidence:.2f}")
                else:
                    rospy.logdebug(f"Failed to get 3D coordinate for detection at ({center_x}, {center_y})")
            
            # 如果有有效的苹果检测结果，选择最近的一个发布
            if valid_apples:
                # 根据距离排序，选择最近的
                closest_apple = min(valid_apples, key=lambda x: x['distance'])
                
                rospy.loginfo_throttle(1.0, 
                    f"Selected closest apple: distance = {closest_apple['distance']:.3f}m, "
                    f"confidence = {closest_apple['confidence']:.2f}")
                
                # 发布最近的苹果坐标
                self.publish_closest_apple(closest_apple)
                
                # 在图像上高亮显示选中的苹果
                self.highlight_selected_apple(display_image, closest_apple)
            
            # 发布带检测框的图像
            self.publish_detection_image(display_image)
            
            # 显示图像
            if self.enable_display:
                cv2.imshow("Apple Detection", display_image)
                cv2.waitKey(1)
                
        except Exception as e:
            rospy.logerr(f"Error in detection processing: {str(e)}")

    def highlight_selected_apple(self, image, selected_apple):
        """在图像上高亮显示选中的苹果"""
        try:
            center_x, center_y = selected_apple['center_2d']
            
            # 绘制更大的红色圆圈表示选中
            cv2.circle(image, (center_x, center_y), 10, (0, 0, 255), 3)
            
            # 添加"SELECTED"标签
            label = "Target"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # 标签位置
            label_x = center_x - label_size[0] // 2
            label_y = center_y - 20
            
            # 确保标签在图像范围内
            label_x = max(0, min(label_x, image.shape[1] - label_size[0]))
            label_y = max(label_size[1], label_y)
            
            # 绘制标签背景
            cv2.rectangle(image, 
                        (label_x - 5, label_y - label_size[1] - 5), 
                        (label_x + label_size[0] + 5, label_y + 5), 
                        (0, 0, 255), -1)
            
            # 绘制标签文字
            cv2.putText(image, label, (label_x, label_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        except Exception as e:
            rospy.logerr(f"Error highlighting selected apple: {str(e)}")

    def draw_detection_result(self, image, detection, coord_3d):
        """在图像上绘制检测结果"""
        try:
            bbox = detection['bbox']
            center_x, center_y = detection['center']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # 边界框坐标
            x1, y1, x2, y2 = map(int, bbox)
            
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制中心点
            cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # 绘制类别和置信度标签
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # 标签背景
            cv2.rectangle(image, 
                        (x1, y1 - label_size[1] - 10), 
                        (x1 + label_size[0], y1), 
                        (0, 255, 0), -1)
            
            # 标签文字
            cv2.putText(image, label, (x1, y1 - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # 绘制3D坐标信息
            if coord_3d:
                x, y, z = coord_3d
                coord_label = f"3D: ({x:.2f}, {y:.2f}, {z:.2f})"
                cv2.putText(image, coord_label, 
                          (x1, y2 + 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                # 显示距离
                distance_label = f"Dist: {z:.2f}m"
                cv2.putText(image, distance_label, 
                          (x1, y2 + 40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
        except Exception as e:
            rospy.logerr(f"Error drawing detection result: {str(e)}")

    def publish_closest_apple(self, apple_data):
        """发布最近的苹果坐标"""
        try:
            # 创建Float32MultiArray消息
            msg = Float32MultiArray()
            
            # 数据格式：[1, detection_id, confidence, center_x_2d, center_y_2d, x_3d, y_3d, z_3d]
            data = [
                1.0,  # 只发布一个检测结果
                float(apple_data['detection_id']),  # 检测ID
                apple_data['confidence'],           # 置信度
                float(apple_data['center_2d'][0]),  # 2D中心点x
                float(apple_data['center_2d'][1]),  # 2D中心点y
                apple_data['center_3d'][0],         # 3D坐标x
                apple_data['center_3d'][1],         # 3D坐标y
                apple_data['center_3d'][2]          # 3D坐标z
            ]
            
            msg.data = data
            self.coordinate_pub.publish(msg)
            
            rospy.logdebug(f"Published closest apple at distance: {apple_data['distance']:.3f}m")
            
        except Exception as e:
            rospy.logerr(f"Error publishing closest apple: {str(e)}")

    def publish_detection_image(self, image):
        """发布带检测框的图像"""
        try:
            # 转换OpenCV图像为ROS消息
            image_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
            image_msg.header.stamp = rospy.Time.now()
            image_msg.header.frame_id = "camera_color_optical_frame"
            
            self.detection_image_pub.publish(image_msg)
            
        except CvBridgeError as e:
            rospy.logerr(f"Error publishing detection image: {str(e)}")
        except Exception as e:
            rospy.logerr(f"Error in image publishing: {str(e)}")

    def is_data_ready(self):
        """检查数据是否准备好"""
        with self.data_lock:
            return (self.color_received and 
                   self.depth_received and 
                   self.camera_info_received)

    def cleanup(self):
        """清理资源"""
        try:
            cv2.destroyAllWindows()
            rospy.loginfo("Apple detection node cleanup completed")
        except Exception as e:
            rospy.logerr(f"Error in cleanup: {str(e)}")

    def run(self):
        """运行主循环"""
        rate = rospy.Rate(30)  # 30 Hz
        
        try:
            while not rospy.is_shutdown():
                # 检查数据状态
                if self.is_data_ready():
                    rospy.logdebug_throttle(5.0, "All data ready!")
                
                if self.yolo_initialized:
                    rospy.logdebug_throttle(10.0, "YOLO detector ready!")
                
                rate.sleep()
                
        except rospy.ROSInterruptException:
            rospy.loginfo("Node interrupted by user")
        except Exception as e:
            rospy.logerr(f"Error in main loop: {str(e)}")
        finally:
            self.cleanup()


def main():
    """主函数"""
    try:
        # 创建苹果检测节点
        apple_detection_node = AppleDetectionNode()
        
        rospy.loginfo("Enhanced apple detection node started!")
        
        # 运行节点
        apple_detection_node.run()
        
    except Exception as e:
        rospy.logerr(f"Failed to start apple detection node: {str(e)}")


if __name__ == '__main__':
    main()