#include <ros/ros.h>
#include <moveit_ctrl/JoystickState.h>
#include <iostream>
#include <iomanip>
#include "Uart_Thread.hpp"

Uart_Thread uart("/dev/ttyUSB0", true, true); // 初始化串口线程，开启读写线程
// Xbox手柄信息结构体
struct XBOX_INFOS
{
    uint8_t left_x;        // 左摇杆X轴 (0-255)
    uint8_t left_y;        // 左摇杆Y轴 (0-255)
    uint8_t right_x;       // 右摇杆X轴 (0-255)
    uint8_t right_y;       // 右摇杆Y轴 (0-255)
    uint8_t trigger_left;  // 左扳机 (0-255)
    uint8_t trigger_right; // 右扳机 (0-255)
    bool button_a;         // 按钮A
    bool button_b;         // 按钮B
    bool button_x;         // 按钮X
    bool button_y;         // 按钮Y
    bool button_lb;        // 左bumper
    bool button_rb;        // 右bumper
};

class XboxChassisController
{
private:
    ros::NodeHandle nh_;
    ros::Subscriber joystick_sub_;
    XBOX_INFOS xbox_data_;

    // 模式状态枚举
    enum ControlMode
    {
        MODE_CHASSIS = 0, // 底盘控制模式（默认）
        MODE_ARM = 1      // 机械臂控制模式
    };

    ControlMode current_mode_; // 当前控制模式

    // 按钮状态记录（用于检测按下事件和防抖）
    bool prev_button_x_;
    bool prev_button_a_;
    bool prev_button_b_;
    bool prev_button_y_;
    bool prev_button_lb_;
    bool prev_button_rb_;

    // 防抖计时器
    ros::Time last_button_x_time_;
    ros::Time last_button_a_time_;
    ros::Time last_button_b_time_;
    ros::Time last_button_y_time_;

    // 防抖延迟时间（秒）
    static constexpr double DEBOUNCE_DELAY = 0.1; // 100ms防抖

public:
    XboxChassisController() : current_mode_(MODE_CHASSIS), // 默认底盘模式
                              prev_button_x_(false), prev_button_a_(false),
                              prev_button_b_(false), prev_button_y_(false),
                              prev_button_lb_(false), prev_button_rb_(false)
    {
        // 初始化xbox数据结构体
        memset(&xbox_data_, 0, sizeof(XBOX_INFOS));

        // 初始化时间
        ros::Time now = ros::Time::now();
        last_button_x_time_ = now;
        last_button_a_time_ = now;
        last_button_b_time_ = now;
        last_button_y_time_ = now;

        // 订阅手柄状态话题
        joystick_sub_ = nh_.subscribe("/joystick_state", 10,
                                      &XboxChassisController::joystickCallback, this);

        ROS_INFO("Xbox Chassis Controller node started");
        ROS_INFO("Press X button to switch between Chassis Mode and Arm Mode");
        printCurrentMode();
    }

    // 将float范围(-1.0到1.0)转换为uint8_t范围(0-255)
    uint8_t floatToUint8(float value)
    {
        // 将-1.0到1.0的范围映射到0-255
        float normalized = (value + 1.0f) / 2.0f;                // 转换为0.0-1.0范围
        normalized = std::max(0.0f, std::min(1.0f, normalized)); // 确保在有效范围内
        return static_cast<uint8_t>(normalized * 255);
    }

    // 将float范围(0.0到1.0)转换为uint8_t范围(0-255) - 用于扳机
    uint8_t triggerFloatToUint8(float value)
    {
        float normalized = std::max(0.0f, std::min(1.0f, value)); // 确保在0.0-1.0范围内
        return static_cast<uint8_t>(normalized * 255);
    }

    // 检测按钮按下事件（带防抖功能）
    bool isButtonPressed(bool current_state, bool &prev_state, ros::Time &last_time, ros::Time current_time)
    {
        bool button_pressed = false;

        // 检测从未按下到按下的状态变化
        if (current_state && !prev_state)
        {
            // 检查防抖延迟
            if ((current_time - last_time).toSec() > DEBOUNCE_DELAY)
            {
                button_pressed = true;
                last_time = current_time;
            }
        }

        prev_state = current_state;
        return button_pressed;
    }

    void printCurrentMode()
    {
        switch (current_mode_)
        {
        case MODE_CHASSIS:
            ROS_INFO("Current Mode: Chassis Control");
            break;
        case MODE_ARM:
            ROS_INFO("Current Mode: Arm Control");
            break;
        }
    }

    void joystickCallback(const moveit_ctrl::JoystickState::ConstPtr &msg)
    {
        ros::Time current_time = ros::Time::now();

        // 转换手柄数据到结构体
        xbox_data_.left_x = floatToUint8(msg->left_x);
        xbox_data_.left_y = floatToUint8(msg->left_y);
        xbox_data_.right_x = floatToUint8(msg->right_x);
        xbox_data_.right_y = floatToUint8(msg->right_y);
        xbox_data_.trigger_left = triggerFloatToUint8(msg->trigger_left);
        xbox_data_.trigger_right = triggerFloatToUint8(msg->trigger_right);
        xbox_data_.button_a = msg->button_a;
        xbox_data_.button_b = msg->button_b;
        xbox_data_.button_x = msg->button_x;
        xbox_data_.button_y = msg->button_y;
        xbox_data_.button_lb = msg->button_lb;
        xbox_data_.button_rb = msg->button_rb;

        // 检测X按钮按下事件（模式切换）
        if (isButtonPressed(xbox_data_.button_x, prev_button_x_, last_button_x_time_, current_time))
        {
            // 在两种模式间切换：底盘模式 <-> 机械臂模式
            switch (current_mode_)
            {
            case MODE_CHASSIS:
                current_mode_ = MODE_ARM;
                break;
            case MODE_ARM:
                current_mode_ = MODE_CHASSIS;
                break;
            }
            printCurrentMode();
        }

        // 检测其他按钮按下事件（带防抖）
        bool button_a_pressed = isButtonPressed(xbox_data_.button_a, prev_button_a_, last_button_a_time_, current_time);
        bool button_b_pressed = isButtonPressed(xbox_data_.button_b, prev_button_b_, last_button_b_time_, current_time);
        bool button_y_pressed = isButtonPressed(xbox_data_.button_y, prev_button_y_, last_button_y_time_, current_time);

        // 处理按钮按下事件
        if (button_a_pressed)
        {
            ROS_INFO("Button A pressed");
            handleButtonA();
        }
        if (button_b_pressed)
        {
            ROS_INFO("Button B pressed");
            handleButtonB();
        }
        if (button_y_pressed)
        {
            ROS_INFO("Button Y pressed");
            handleButtonY();
        }

        // 更新LB和RB按钮状态（如果需要）
        prev_button_lb_ = xbox_data_.button_lb;
        prev_button_rb_ = xbox_data_.button_rb;

        // 打印手柄数据（可选）
        printXboxData();

        // 根据当前模式执行相应操作
        switch (current_mode_)
        {
        case MODE_CHASSIS:
            executeChassisControl();
            break;
        case MODE_ARM:
            executeArmControl();
            break;
        }
    }

    void printXboxData()
    {
        // 每隔一定时间打印数据，避免刷屏
        static ros::Time last_print_time = ros::Time::now();
        ros::Time current_time = ros::Time::now();

        if ((current_time - last_print_time).toSec() > 0.1)
        { // 每0.1秒打印一次，提升响应速度
            std::cout << "\n=== Xbox Controller Data ===" << std::endl;
            std::cout << "Left Stick: X=" << std::setw(3) << (int)xbox_data_.left_x
                      << " Y=" << std::setw(3) << (int)xbox_data_.left_y << std::endl;
            std::cout << "Right Stick: X=" << std::setw(3) << (int)xbox_data_.right_x
                      << " Y=" << std::setw(3) << (int)xbox_data_.right_y << std::endl;
            std::cout << "Triggers: L=" << std::setw(3) << (int)xbox_data_.trigger_left
                      << " R=" << std::setw(3) << (int)xbox_data_.trigger_right << std::endl;
            std::cout << "Buttons: A=" << xbox_data_.button_a << " B=" << xbox_data_.button_b
                      << " X=" << xbox_data_.button_x << " Y=" << xbox_data_.button_y << std::endl;
            std::cout << "Bumpers: LB=" << xbox_data_.button_lb << " RB=" << xbox_data_.button_rb << std::endl;

            // 显示当前模式
            const char *mode_str;
            switch (current_mode_)
            {
            case MODE_CHASSIS:
                mode_str = "Chassis Control";
                break;
            case MODE_ARM:
                mode_str = "Arm Control";
                break;
            default:
                mode_str = "Unknown";
                break;
            }
            std::cout << "Current Mode: " << mode_str << std::endl;

            last_print_time = current_time;
        }
    }

    // 按钮A处理函数
    void handleButtonA()
    {
        switch (current_mode_)
        {
        case MODE_CHASSIS:
            ROS_INFO("Chassis Mode - Button A:");
            // 添加底盘相关的A按钮功能
            break;
        case MODE_ARM:
            ROS_INFO("Arm Mode - Button A");
            // 添加机械臂相关的A按钮功能
            break;
        }
    }

    // 按钮B处理函数
    void handleButtonB()
    {
        switch (current_mode_)
        {
        case MODE_CHASSIS:
            ROS_INFO("Chassis Mode - Button Bn");
            // 添加底盘相关的B按钮功能
            break;
        case MODE_ARM:
            ROS_INFO("Arm Mode - Button B");
            // 添加机械臂相关的B按钮功能
            break;
        }
    }

    // 按钮Y处理函数
    void handleButtonY()
    {
        switch (current_mode_)
        {
        case MODE_CHASSIS:
            ROS_INFO("Chassis Mode - Button Y");
            // 添加底盘相关的Y按钮功能
            break;
        case MODE_ARM:
            ROS_INFO("Arm Mode - Button Y");
            // 添加机械臂相关的Y按钮功能
            break;
        }
    }

    void executeChassisControl()
    {
        // 底盘控制逻辑
        // TODO: 实现底盘控制
        //串口发送底盘数据
        uart.Mission_Send(Uart_Thread_Space::Chassis_Infos_Send, (uint8_t)xbox_data_.left_x, (uint8_t)xbox_data_.left_y, (uint8_t)xbox_data_.right_x, (uint8_t)xbox_data_.right_y);
    }

    void executeArmControl()
    {
        // 机械臂控制逻辑
        // TODO: 实现机械臂控制
        /*
        // 机械臂控制代码
        // 使用摇杆控制机械臂各轴运动
        // 使用扳机控制夹爪
        // 使用LB/RB按钮实现预设位置等功能
        // A/B/Y按钮功能已在handleButtonA/B/Y中处理
        */
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "xbox_chassis");
    // Uart_Thread uart("/dev/ttyUSB0", true, true); // 初始化串口线程，开启读写线程
    try
    {
        XboxChassisController controller;

        ROS_INFO("xbox_chassis node started");

        ROS_INFO("xbox_chassis node running...");
        ros::spin();
    }
    catch (const std::exception &e)
    {
        ROS_ERROR("Node execution error: %s", e.what());
        return -1;
    }

    ROS_INFO("xbox_chassis node exited");
    return 0;
}