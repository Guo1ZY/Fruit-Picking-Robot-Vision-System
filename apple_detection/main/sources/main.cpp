/**
 * @file main.cpp
 * @author Guo1ZY（1352872047@qq.com)
 * @brief Enhanced with Apple Detection and 3D Coordinate Publishing
 * @version 0.2
 * @date 2025-06-28
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PointStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <opencv2/imgproc.hpp>
#include "Yolov8.hpp"

#define MIN_DISTANCE 0.15     // 最小距离，单位：米
#define MAX_DISTANCE 0.8      // 最大距离，单位：米
#define image_width 640       // 图像宽度
#define image_height 400      // 图像高度
#define DEPTH_FILTER_RADIUS 2 // 深度滤波半径

using namespace std;

/*********************************声明************************************ */
/*训练好的模型路径*/
const string model_Path = "/home/zy/ws_livox/src/apple_detection/yolo/yolov8/model/best.onnx";

/*分类标签路径*/
const string clasesPath = "/home/zy/ws_livox/src/apple_detection/yolo/yolov8/model/classes.txt";

// /*yolo 声明*/
Yolov8 yolo(model_Path, clasesPath, false, cv::Size(256, 256), 0.8, 0.6);
/*********************************类定义初始化************************************ */

class CameraSubscriber
{
private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;

    // 订阅器
    image_transport::Subscriber color_sub_;
    image_transport::Subscriber depth_sub_;
    ros::Subscriber pointcloud_sub_;
    ros::Subscriber color_info_sub_;
    ros::Subscriber depth_info_sub_;

    // 发布器
    ros::Publisher coordinate_pub_; // 3D坐标发布器

    // 存储当前图像数据
    cv::Mat current_color_image_;
    cv::Mat current_depth_image_;
    cv::Mat aligned_depth_image_; // 对齐后的深度图像
    cv::Mat aligned_color_image_; // 对齐后的彩色图像
    sensor_msgs::PointCloud2::ConstPtr current_pointcloud_;

    // 相机参数
    sensor_msgs::CameraInfo color_camera_info_;
    sensor_msgs::CameraInfo depth_camera_info_;

    // 相机内参矩阵
    cv::Mat color_K_, depth_K_;
    cv::Mat color_D_, depth_D_; // 畸变参数

    bool color_received_;
    bool depth_received_;
    bool pointcloud_received_;
    bool camera_info_received_;

public:
    CameraSubscriber() : it_(nh_),
                         color_received_(false), depth_received_(false),
                         pointcloud_received_(false), camera_info_received_(false)
    {
        // 订阅彩色图像
        color_sub_ = it_.subscribe("/camera/color/image_raw", 1,
                                   &CameraSubscriber::colorImageCallback, this);

        // 订阅深度图像
        depth_sub_ = it_.subscribe("/camera/depth/image_raw", 1,
                                   &CameraSubscriber::depthImageCallback, this);

        // 订阅点云数据
        pointcloud_sub_ = nh_.subscribe("/camera/depth/points", 1,
                                        &CameraSubscriber::pointCloudCallback, this);

        // 订阅相机信息
        color_info_sub_ = nh_.subscribe("/camera/color/camera_info", 1,
                                        &CameraSubscriber::colorInfoCallback, this);
        depth_info_sub_ = nh_.subscribe("/camera/depth/camera_info", 1,
                                        &CameraSubscriber::depthInfoCallback, this);

        // 初始化发布器
        coordinate_pub_ = nh_.advertise<geometry_msgs::PointStamped>("/coordinate_result", 10);

        ROS_INFO("Camera subscriber initialized!");
        ROS_INFO("Waiting for camera data...");
    }

    // 彩色图像回调函数
    void colorImageCallback(const sensor_msgs::ImageConstPtr &msg)
    {
        try
        {
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            current_color_image_ = cv_ptr->image.clone();
            color_received_ = true;

            ROS_INFO_THROTTLE(1.0, "Received color image: %dx%d",
                              current_color_image_.cols, current_color_image_.rows);

            // 处理彩色图像（包含YOLO检测）
            processColorImage();
        }
        catch (cv_bridge::Exception &e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
    }

    // 深度图像回调函数
    void depthImageCallback(const sensor_msgs::ImageConstPtr &msg)
    {
        try
        {
            cv_bridge::CvImagePtr cv_ptr;
            if (msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1)
            {
                cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
            }
            else if (msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
            {
                cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
            }
            else
            {
                cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
            }

            current_depth_image_ = cv_ptr->image.clone();
            depth_received_ = true;

            ROS_INFO_THROTTLE(1.0, "Received depth image: %dx%d, encoding: %s",
                              current_depth_image_.cols, current_depth_image_.rows, msg->encoding.c_str());

            // 处理深度图像
            processDepthImage();
        }
        catch (cv_bridge::Exception &e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
    }

    // 点云回调函数
    void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        current_pointcloud_ = msg;
        pointcloud_received_ = true;

        // 处理点云数据
        processPointCloud();
    }

    // 彩色相机信息回调函数
    void colorInfoCallback(const sensor_msgs::CameraInfo::ConstPtr &msg)
    {
        color_camera_info_ = *msg;

        // 提取内参矩阵
        color_K_ = (cv::Mat_<double>(3, 3) << msg->K[0], msg->K[1], msg->K[2],
                    msg->K[3], msg->K[4], msg->K[5],
                    msg->K[6], msg->K[7], msg->K[8]);

        // 提取畸变参数
        if (!msg->D.empty())
        {
            color_D_ = cv::Mat(msg->D.size(), 1, CV_64F);
            for (size_t i = 0; i < msg->D.size(); ++i)
            {
                color_D_.at<double>(i) = msg->D[i];
            }
        }

        camera_info_received_ = true;
        ROS_INFO_ONCE("Received color camera info: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f",
                      msg->K[0], msg->K[4], msg->K[2], msg->K[5]);
    }

    // 深度相机信息回调函数
    void depthInfoCallback(const sensor_msgs::CameraInfo::ConstPtr &msg)
    {
        depth_camera_info_ = *msg;

        // 提取内参矩阵
        depth_K_ = (cv::Mat_<double>(3, 3) << msg->K[0], msg->K[1], msg->K[2],
                    msg->K[3], msg->K[4], msg->K[5],
                    msg->K[6], msg->K[7], msg->K[8]);

        // 提取畸变参数
        if (!msg->D.empty())
        {
            depth_D_ = cv::Mat(msg->D.size(), 1, CV_64F);
            for (size_t i = 0; i < msg->D.size(); ++i)
            {
                depth_D_.at<double>(i) = msg->D[i];
            }
        }

        ROS_INFO_ONCE("Received depth camera info: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f",
                      msg->K[0], msg->K[4], msg->K[2], msg->K[5]);
    }

    // 深度图像与彩色图像对齐
    void alignDepthToColor()
    {
        if (current_color_image_.empty() || current_depth_image_.empty() || !camera_info_received_)
        {
            return;
        }

        // 获取图像尺寸
        int color_width = current_color_image_.cols;
        int color_height = current_color_image_.rows;
        int depth_width = current_depth_image_.cols;
        int depth_height = current_depth_image_.rows;

        // 创建对齐后的深度图像
        aligned_depth_image_ = cv::Mat::zeros(color_height, color_width, current_depth_image_.type());

        // 提取相机内参
        double fx_color = color_K_.at<double>(0, 0);
        double fy_color = color_K_.at<double>(1, 1);
        double cx_color = color_K_.at<double>(0, 2);
        double cy_color = color_K_.at<double>(1, 2);

        double fx_depth = depth_K_.at<double>(0, 0);
        double fy_depth = depth_K_.at<double>(1, 1);
        double cx_depth = depth_K_.at<double>(0, 2);
        double cy_depth = depth_K_.at<double>(1, 2);

        // 遍历彩色图像的每个像素
        for (int v = 0; v < color_height; ++v)
        {
            for (int u = 0; u < color_width; ++u)
            {
                // 如果深度和彩色分辨率相同，可以直接复制
                if (depth_width == color_width && depth_height == color_height)
                {
                    if (current_depth_image_.type() == CV_16UC1)
                    {
                        aligned_depth_image_.at<uint16_t>(v, u) = current_depth_image_.at<uint16_t>(v, u);
                    }
                    else if (current_depth_image_.type() == CV_32FC1)
                    {
                        aligned_depth_image_.at<float>(v, u) = current_depth_image_.at<float>(v, u);
                    }
                }
                else
                {
                    // 不同分辨率时的映射
                    int depth_u = (int)(u * depth_width / (double)color_width);
                    int depth_v = (int)(v * depth_height / (double)color_height);

                    if (depth_u >= 0 && depth_u < depth_width && depth_v >= 0 && depth_v < depth_height)
                    {
                        if (current_depth_image_.type() == CV_16UC1)
                        {
                            aligned_depth_image_.at<uint16_t>(v, u) = current_depth_image_.at<uint16_t>(depth_v, depth_u);
                        }
                        else if (current_depth_image_.type() == CV_32FC1)
                        {
                            aligned_depth_image_.at<float>(v, u) = current_depth_image_.at<float>(depth_v, depth_u);
                        }
                    }
                }
            }
        }

        ROS_INFO_THROTTLE(2.0, "Depth image aligned to color image");
    }

    // 深度滤波函数：计算中心点周围正负2个像素的深度值平均
    float getFilteredDepth(int center_u, int center_v)
    {
        if (aligned_depth_image_.empty())
        {
            return 0.0f;
        }

        std::vector<float> valid_depths;

        // 遍历中心点周围正负2个像素
        for (int v = center_v - DEPTH_FILTER_RADIUS; v <= center_v + DEPTH_FILTER_RADIUS; ++v)
        {
            for (int u = center_u - DEPTH_FILTER_RADIUS; u <= center_u + DEPTH_FILTER_RADIUS; ++u)
            {
                // 检查边界
                if (u >= 0 && u < aligned_depth_image_.cols && v >= 0 && v < aligned_depth_image_.rows)
                {
                    float depth_value = 0.0f;

                    // 获取深度值
                    if (aligned_depth_image_.type() == CV_16UC1)
                    {
                        uint16_t depth_raw = aligned_depth_image_.at<uint16_t>(v, u);
                        depth_value = depth_raw / 1000.0f; // 毫米转米
                    }
                    else if (aligned_depth_image_.type() == CV_32FC1)
                    {
                        depth_value = aligned_depth_image_.at<float>(v, u);
                    }

                    // 只考虑有效深度值
                    if (depth_value > 0.0f && depth_value <= 10.0f)
                    {
                        valid_depths.push_back(depth_value);
                    }
                }
            }
        }

        // 计算平均深度
        if (!valid_depths.empty())
        {
            float sum = 0.0f;
            for (float depth : valid_depths)
            {
                sum += depth;
            }
            return sum / valid_depths.size();
        }

        return 0.0f;
    }

    // 根据滤波后的深度值计算3D坐标
    cv::Point3f getFiltered3DCoordinate(int u, int v)
    {
        cv::Point3f point_3d(0, 0, 0);

        if (!camera_info_received_)
        {
            return point_3d;
        }

        // 获取滤波后的深度值
        float depth_value = getFilteredDepth(u, v);

        if (depth_value <= 0.0f)
        {
            return point_3d;
        }

        // 相机内参
        double fx = color_K_.at<double>(0, 0);
        double fy = color_K_.at<double>(1, 1);
        double cx = color_K_.at<double>(0, 2);
        double cy = color_K_.at<double>(1, 2);

        // 计算3D坐标
        point_3d.x = (u - cx) * depth_value / fx;
        point_3d.y = (v - cy) * depth_value / fy;
        point_3d.z = depth_value;

        return point_3d;
    }

    // 处理检测结果并发布最近的苹果坐标
    void processDetectionResults(const std::vector<YoloDetect> &results)
    {
        if (results.empty())
        {
            return;
        }

        std::vector<cv::Point3f> apple_coordinates;
        cv::Mat display_image = current_color_image_.clone();

        // 遍历每个检测结果
        for (const auto &result : results)
        {
            // 检查是否为苹果
            if (result.class_name == "apple")
            {
                // 计算检测框中心点
                int center_u = result.box.x + result.box.width / 2;
                int center_v = result.box.y + result.box.height / 2;

                // 获取滤波后的3D坐标
                cv::Point3f apple_3d = getFiltered3DCoordinate(center_u, center_v);

                if (apple_3d.z > 0.0f) // 有效深度
                {
                    apple_coordinates.push_back(apple_3d);

                    // 在图像上标注中心点和3D坐标
                    cv::circle(display_image, cv::Point(center_u, center_v), 5, cv::Scalar(0, 255, 0), -1);

                    // 显示3D坐标信息
                    std::string coord_text = cv::format("(%.2f, %.2f, %.2f)", apple_3d.x, apple_3d.y, apple_3d.z);
                    cv::putText(display_image, coord_text,
                                cv::Point(center_u - 50, center_v - 20),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

                    ROS_INFO("Apple detected at 3D coordinate: (%.3f, %.3f, %.3f)",
                             apple_3d.x, apple_3d.y, apple_3d.z);
                }
            }
        }

        // 绘制检测结果
        yolo.drawResult(display_image, results);

        // 显示结果图像
        cv::imshow("Apple Detection Result", display_image);
        cv::waitKey(1);

        // 发布最近的苹果坐标
        if (!apple_coordinates.empty())
        {
            // 找到最近的苹果（Z坐标最小）
            cv::Point3f closest_apple = apple_coordinates[0];
            for (const auto &coord : apple_coordinates)
            {
                if (coord.z < closest_apple.z)
                {
                    closest_apple = coord;
                }
            }

            // 发布3D坐标
            geometry_msgs::PointStamped coordinate_msg;
            coordinate_msg.header.stamp = ros::Time::now();
            coordinate_msg.header.frame_id = "camera_link"; // 根据你的相机坐标系名称修改
            coordinate_msg.point.x = closest_apple.x;
            coordinate_msg.point.y = closest_apple.y;
            coordinate_msg.point.z = closest_apple.z;

            coordinate_pub_.publish(coordinate_msg);

            ROS_INFO("Published closest apple coordinate: (%.3f, %.3f, %.3f)",
                     closest_apple.x, closest_apple.y, closest_apple.z);
        }
    }

    // 从对齐的深度图获取3D点云
    std::vector<cv::Point3f> getAligned3DPoints()
    {
        std::vector<cv::Point3f> points_3d;

        if (aligned_depth_image_.empty() || current_color_image_.empty() || !camera_info_received_)
        {
            return points_3d;
        }

        // 相机内参
        double fx = color_K_.at<double>(0, 0);
        double fy = color_K_.at<double>(1, 1);
        double cx = color_K_.at<double>(0, 2);
        double cy = color_K_.at<double>(1, 2);

        int width = current_color_image_.cols;
        int height = current_color_image_.rows;

        for (int v = 0; v < height; ++v)
        {
            for (int u = 0; u < width; ++u)
            {
                float depth_value = 0.0f;

                // 获取深度值
                if (aligned_depth_image_.type() == CV_16UC1)
                {
                    uint16_t depth_raw = aligned_depth_image_.at<uint16_t>(v, u);
                    depth_value = depth_raw / 1000.0f; // 假设深度单位是毫米，转换为米
                }
                else if (aligned_depth_image_.type() == CV_32FC1)
                {
                    depth_value = aligned_depth_image_.at<float>(v, u);
                }

                // 跳过无效深度值
                if (depth_value <= 0.0f || depth_value > 10.0f)
                    continue;

                // 像素坐标转3D坐标
                float x = (u - cx) * depth_value / fx;
                float y = (v - cy) * depth_value / fy;
                float z = depth_value;

                points_3d.push_back(cv::Point3f(x, y, z));
            }
        }

        return points_3d;
    }

    // 获取指定像素的3D坐标
    cv::Point3f getPixel3DCoordinate(int u, int v)
    {
        cv::Point3f point_3d(0, 0, 0);

        if (aligned_depth_image_.empty() || !camera_info_received_)
        {
            return point_3d;
        }

        if (u < 0 || u >= aligned_depth_image_.cols || v < 0 || v >= aligned_depth_image_.rows)
        {
            return point_3d;
        }

        // 相机内参
        double fx = color_K_.at<double>(0, 0);
        double fy = color_K_.at<double>(1, 1);
        double cx = color_K_.at<double>(0, 2);
        double cy = color_K_.at<double>(1, 2);

        float depth_value = 0.0f;

        // 获取深度值
        if (aligned_depth_image_.type() == CV_16UC1)
        {
            uint16_t depth_raw = aligned_depth_image_.at<uint16_t>(v, u);
            depth_value = depth_raw / 1000.0f; // 毫米转米
        }
        else if (aligned_depth_image_.type() == CV_32FC1)
        {
            depth_value = aligned_depth_image_.at<float>(v, u);
        }

        if (depth_value > 0.0f && depth_value <= 10.0f)
        {
            point_3d.x = (u - cx) * depth_value / fx;
            point_3d.y = (v - cy) * depth_value / fy;
            point_3d.z = depth_value;
        }

        return point_3d;
    }

    // 处理彩色图像（添加YOLO检测）
    void processColorImage()
    {
        if (current_color_image_.empty())
            return;

        // 只有在对齐数据准备好时才进行检测
        if (isAlignedDataReady())
        {
            // 使用YOLO进行检测
            std::vector<YoloDetect> detection_results = yolo.detect(current_color_image_);

            // 处理检测结果
            if (!detection_results.empty())
            {
                ROS_INFO("Detected %lu objects", detection_results.size());
                processDetectionResults(detection_results);
            }
        }
        else
        {
            // 如果对齐数据未准备好，仅显示彩色图像
            cv::imshow("Color Image", current_color_image_);
            cv::waitKey(1);
        }
    }

    // 处理深度图像
    void processDepthImage()
    {
        if (current_depth_image_.empty())
            return;

        // 如果彩色图像也已接收，进行对齐
        if (!current_color_image_.empty() && camera_info_received_)
        {
            alignDepthToColor();
        }

        // 深度图像可视化
        cv::Mat depth_display;
        if (current_depth_image_.type() == CV_16UC1)
        {
            // 16位深度图转换为8位显示
            current_depth_image_.convertTo(depth_display, CV_8UC1, 255.0 / 65535.0);
        }
        else if (current_depth_image_.type() == CV_32FC1)
        {
            // 32位浮点深度图转换为8位显示
            cv::normalize(current_depth_image_, depth_display, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        }

        // cv::imshow("Depth Image", depth_display);
        // cv::waitKey(1);
    }

    // 可视化对齐后的深度图
    void visualizeAlignedDepth(cv::Mat &output)
    {
        if (aligned_depth_image_.empty())
            return;

        if (aligned_depth_image_.type() == CV_16UC1)
        {
            aligned_depth_image_.convertTo(output, CV_8UC1, 255.0 / 65535.0);
        }
        else if (aligned_depth_image_.type() == CV_32FC1)
        {
            cv::normalize(aligned_depth_image_, output, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        }

        // 转换为彩色图以便叠加
        cv::applyColorMap(output, output, cv::COLORMAP_JET);
    }

    // 处理点云数据
    void processPointCloud()
    {
        if (!current_pointcloud_)
            return;

        // 在这里添加你的点云处理逻辑
        // 例如：3D重建、物体识别等

        // 示例：打印点云基本信息
        ROS_INFO_THROTTLE(5.0, "Point cloud frame: %s, timestamp: %.3f",
                          current_pointcloud_->header.frame_id.c_str(),
                          current_pointcloud_->header.stamp.toSec());
    }

    // 检查数据是否准备好
    bool isDataReady() const
    {
        return color_received_ && depth_received_ && camera_info_received_;
    }

    // 检查对齐数据是否准备好
    bool isAlignedDataReady() const
    {
        return !aligned_depth_image_.empty() && !current_color_image_.empty();
    }

    // 获取图像数据（供外部使用）
    cv::Mat getCurrentColorImage() const { return current_color_image_.clone(); }
    cv::Mat getCurrentDepthImage() const { return current_depth_image_.clone(); }
    cv::Mat getAlignedDepthImage() const { return aligned_depth_image_.clone(); }
    sensor_msgs::PointCloud2::ConstPtr getCurrentPointCloud() const { return current_pointcloud_; }

    // 获取相机参数
    sensor_msgs::CameraInfo getColorCameraInfo() const { return color_camera_info_; }
    sensor_msgs::CameraInfo getDepthCameraInfo() const { return depth_camera_info_; }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "camera_subscriber_node");

    CameraSubscriber camera_subscriber;

    ROS_INFO("Camera subscriber node started!");
    ROS_INFO("Apple detection and 3D coordinate publishing enabled!");

    ros::Rate rate(30); // 30 Hz

    while (ros::ok())
    {
        ros::spinOnce();

        // 检查数据是否准备好
        if (camera_subscriber.isDataReady())
        {
            // ROS_INFO("Data ready!");
        }

        rate.sleep();
    }

    cv::destroyAllWindows();
    return 0;
}