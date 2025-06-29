#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@file enhanced_coordinate_listener.py
@author Guo1ZY
@brief Enhanced test node to listen to closest apple detection coordinates
@version 0.2
@date 2025-06-29
"""

import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class EnhancedCoordinateListener:
    def __init__(self):
        """初始化增强版坐标监听器"""
        rospy.init_node('enhanced_coordinate_listener', anonymous=True)
        
        self.bridge = CvBridge()
        
        # 统计信息
        self.total_detections = 0
        self.valid_detections = 0
        self.detection_history = []
        self.max_history = 100
        
        # 订阅检测结果
        self.coordinate_sub = rospy.Subscriber(
            '/apple_detection/coordinates', 
            Float32MultiArray, 
            self.coordinate_callback, 
            queue_size=10
        )
        
        # 订阅检测图像
        self.image_sub = rospy.Subscriber(
            '/apple_detection/image', 
            Image, 
            self.image_callback, 
            queue_size=1
        )
        
        rospy.loginfo("Enhanced coordinate listener started!")
        rospy.loginfo("Listening for closest apple detection results...")
        rospy.loginfo("System expects only the closest apple to be published.")

    def coordinate_callback(self, msg):
        """坐标数据回调函数"""
        try:
            data = msg.data
            if len(data) < 1:
                return
            
            num_detections = int(data[0])
            self.total_detections += 1
            
            if num_detections == 0:
                rospy.loginfo("No apple detections in this frame")
                return
            
            if num_detections != 1:
                rospy.logwarn(f"Expected 1 detection (closest apple), but received {num_detections}")
            
            # 解析检测结果（应该只有一个最近的苹果）
            if len(data) >= 8:  # 1 + 7个检测数据
                self.valid_detections += 1
                
                detection_id = int(data[1])
                confidence = data[2]
                center_x_2d = int(data[3])
                center_y_2d = int(data[4])
                x_3d = data[5]
                y_3d = data[6]
                z_3d = data[7]
                
                # 计算欧几里得距离
                euclidean_distance = np.sqrt(x_3d**2 + y_3d**2 + z_3d**2)
                
                # 存储检测历史
                detection_record = {
                    'timestamp': rospy.Time.now().to_sec(),
                    'confidence': confidence,
                    'distance': z_3d,
                    'euclidean_distance': euclidean_distance,
                    'position_3d': (x_3d, y_3d, z_3d),
                    'position_2d': (center_x_2d, center_y_2d)
                }
                
                self.detection_history.append(detection_record)
                if len(self.detection_history) > self.max_history:
                    self.detection_history.pop(0)
                
                # 显示详细信息
                rospy.loginfo("=" * 60)
                rospy.loginfo(f"CLOSEST APPLE DETECTION #{self.valid_detections}")
                rospy.loginfo("=" * 60)
                rospy.loginfo(f"Detection ID: {detection_id}")
                rospy.loginfo(f"Confidence: {confidence:.3f}")
                rospy.loginfo(f"2D Center: ({center_x_2d}, {center_y_2d}) pixels")
                rospy.loginfo(f"3D Position: ({x_3d:.3f}, {y_3d:.3f}, {z_3d:.3f}) meters")
                rospy.loginfo(f"Distance (Z-axis): {z_3d:.3f} meters")
                rospy.loginfo(f"Euclidean Distance: {euclidean_distance:.3f} meters")
                
                # 计算相对于相机的角度
                angle_x = np.arctan2(x_3d, z_3d) * 180 / np.pi  # 水平角度
                angle_y = np.arctan2(y_3d, z_3d) * 180 / np.pi  # 垂直角度
                rospy.loginfo(f"Relative Angles: X={angle_x:.1f}°, Y={angle_y:.1f}°")
                
                # 显示统计信息
                if len(self.detection_history) > 1:
                    self.display_statistics()
                
                rospy.loginfo("=" * 60)
                
            else:
                rospy.logerr(f"Invalid data length: expected 8, got {len(data)}")
                
        except Exception as e:
            rospy.logerr(f"Error processing coordinate data: {str(e)}")

    def display_statistics(self):
        """显示统计信息"""
        try:
            if len(self.detection_history) < 2:
                return
            
            # 计算统计数据
            recent_detections = self.detection_history[-10:]  # 最近10次检测
            
            distances = [d['distance'] for d in recent_detections]
            confidences = [d['confidence'] for d in recent_detections]
            
            avg_distance = np.mean(distances)
            std_distance = np.std(distances)
            avg_confidence = np.mean(confidences)
            
            min_distance = min(distances)
            max_distance = max(distances)
            
            rospy.loginfo("STATISTICS (Last 10 detections):")
            rospy.loginfo(f"  Average Distance: {avg_distance:.3f} ± {std_distance:.3f} m")
            rospy.loginfo(f"  Distance Range: {min_distance:.3f} - {max_distance:.3f} m")
            rospy.loginfo(f"  Average Confidence: {avg_confidence:.3f}")
            rospy.loginfo(f"  Total Valid Detections: {self.valid_detections}")
            
            # 检查稳定性
            if std_distance < 0.05:  # 5cm标准差
                rospy.loginfo("  Status: STABLE detection")
            elif std_distance < 0.1:  # 10cm标准差
                rospy.loginfo("  Status: MODERATELY STABLE detection")
            else:
                rospy.loginfo("  Status: UNSTABLE detection")
                
        except Exception as e:
            rospy.logerr(f"Error calculating statistics: {str(e)}")

    def image_callback(self, msg):
        """检测图像回调函数"""
        try:
            # 转换并显示图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 在图像上添加统计信息
            self.overlay_statistics(cv_image)
            
            cv2.imshow("Enhanced Apple Detection Results", cv_image)
            key = cv2.waitKey(1) & 0xFF
            
            # 按's'保存当前帧
            if key == ord('s'):
                timestamp = rospy.Time.now().to_sec()
                filename = f"/tmp/apple_detection_{timestamp:.0f}.jpg"
                cv2.imwrite(filename, cv_image)
                rospy.loginfo(f"Saved image: {filename}")
            
            # 按'r'重置统计
            elif key == ord('r'):
                self.reset_statistics()
                rospy.loginfo("Statistics reset")
            
        except Exception as e:
            rospy.logerr(f"Error processing detection image: {str(e)}")

    def overlay_statistics(self, image):
        """在图像上叠加统计信息"""
        try:
            if len(self.detection_history) == 0:
                return
            
            # 获取最新检测数据
            latest = self.detection_history[-1]
            
            # 准备显示文本
            texts = [
                f"Total Detections: {self.valid_detections}",
                f"Confidence: {latest['confidence']:.3f}",
                f"Distance: {latest['distance']:.3f}m",
                f"Position: ({latest['position_3d'][0]:.2f}, {latest['position_3d'][1]:.2f}, {latest['position_3d'][2]:.2f})",
            ]
            
            # 如果有足够的历史数据，显示平均值
            if len(self.detection_history) >= 5:
                recent = self.detection_history[-5:]
                avg_dist = np.mean([d['distance'] for d in recent])
                texts.append(f"Avg Distance (5): {avg_dist:.3f}m")
            
            # 在图像左上角显示文本
            y_offset = 30
            for i, text in enumerate(texts):
                y = y_offset + i * 25
                
                # 文本背景
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(image, (10, y - 20), (20 + text_size[0], y + 5), (0, 0, 0), -1)
                
                # 文本
                cv2.putText(image, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 显示控制说明
            help_text = "Press 's' to save, 'r' to reset, 'q' to quit"
            text_size = cv2.getTextSize(help_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            y_help = image.shape[0] - 10
            cv2.rectangle(image, (10, y_help - 20), (20 + text_size[0], y_help + 5), (0, 0, 0), -1)
            cv2.putText(image, help_text, (15, y_help), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        except Exception as e:
            rospy.logerr(f"Error overlaying statistics: {str(e)}")

    def reset_statistics(self):
        """重置统计信息"""
        self.total_detections = 0
        self.valid_detections = 0
        self.detection_history = []

    def run(self):
        """运行监听器"""
        try:
            rospy.loginfo("Enhanced coordinate listener running...")
            rospy.loginfo("Controls:")
            rospy.loginfo("  's' - Save current image")
            rospy.loginfo("  'r' - Reset statistics")
            rospy.loginfo("  'q' - Quit (or Ctrl+C)")
            
            rospy.spin()
            
        except rospy.ROSInterruptException:
            rospy.loginfo("Enhanced coordinate listener interrupted")
        finally:
            cv2.destroyAllWindows()
            
            # 显示最终统计
            if self.valid_detections > 0:
                rospy.loginfo("=" * 60)
                rospy.loginfo("FINAL STATISTICS")
                rospy.loginfo("=" * 60)
                rospy.loginfo(f"Total Valid Detections: {self.valid_detections}")
                
                if len(self.detection_history) > 0:
                    distances = [d['distance'] for d in self.detection_history]
                    confidences = [d['confidence'] for d in self.detection_history]
                    
                    rospy.loginfo(f"Average Distance: {np.mean(distances):.3f} ± {np.std(distances):.3f} m")
                    rospy.loginfo(f"Distance Range: {min(distances):.3f} - {max(distances):.3f} m")
                    rospy.loginfo(f"Average Confidence: {np.mean(confidences):.3f}")
                    
                rospy.loginfo("=" * 60)

def main():
    """主函数"""
    try:
        listener = EnhancedCoordinateListener()
        listener.run()
    except Exception as e:
        rospy.logerr(f"Failed to start enhanced coordinate listener: {str(e)}")

if __name__ == '__main__':
    main()