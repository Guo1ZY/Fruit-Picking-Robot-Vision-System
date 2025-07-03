#!/usr/bin/env python

import rospy
import time
import random
from moveit_ctrl.srv import JointMoveitCtrl, JointMoveitCtrlRequest
from tf.transformations import quaternion_from_euler
from std_msgs.msg import String


import struct
import time
import threading
from collections import defaultdict

from moveit_ctrl.msg import JoystickState

# 末端姿态全局变量（欧拉角）

global_endpose_euler = [0.331014, 0, 0.3518909, -0.0052452780065936, 1.5605301318390152, -0.0107128036906411]
# 全局变量 - 轴数据
global_left_x = 0.0      # 左摇杆X轴 (code 0)
global_left_y = 0.0      # 左摇杆Y轴 (code 1)
global_right_x = 0.0     # 右摇杆X轴 (code 2)
global_right_y = 0.0     # 右摇杆Y轴 (code 5)
global_trigger_left = 0.0  # 左扳机 (code 10)
global_trigger_right = 0.0 # 右扳机 (code 9)

# 全局变量 - 按钮数据
global_button_a = False      # 按钮A (code 304)
global_button_b = False      # 按钮B (code 305)
global_button_x = False      # 按钮X (code 307)
global_button_y = False      # 按钮Y (code 308)
global_button_lb = False     # 左 bumper (code 310)
global_button_rb = False     # 右 bumper (code 311)


# 线程锁，确保数据一致性
data_lock = threading.Lock()

# 机械臂控制相关全局变量
global_arm_running = False   # 机械臂控制线程运行标志
global_arm_thread = None     # 机械臂控制线程对象

# 手柄轴和按钮映射配置
AXIS_CONFIG = {
    0: {"name": "LEFT_X", "min": 0, "max": 65535, "deadzone": 0.1},    # 左摇杆 X 轴 (code 0)
    1: {"name": "LEFT_Y", "min": 0, "max": 65535, "deadzone": 0.1},    # 左摇杆 Y 轴 (code 1)
    2: {"name": "RIGHT_X", "min": 0, "max": 65535, "deadzone": 0.1},   # 右摇杆 X 轴 (code 2)
    5: {"name": "RIGHT_Y", "min": 0, "max": 65535, "deadzone": 0.1},   # 右摇杆 Y 轴 (code 5)
    10: {"name": "TRIGGER_LEFT", "min": 0, "max": 1024, "deadzone": 0.1},  # 左扳机 (code 10)
    9: {"name": "TRIGGER_RIGHT", "min": 0, "max": 1024, "deadzone": 0.1},  # 右扳机 (code 9)
}


BUTTON_CONFIG = {
    304: {"name": "BUTTON_A", "type": "binary"},
    305: {"name": "BUTTON_B", "type": "binary"},
    307: {"name": "BUTTON_X", "type": "binary"},
    308: {"name": "BUTTON_Y", "type": "binary"},
    310: {"name": "BUTTON_LB", "type": "binary"},
    311: {"name": "BUTTON_RB", "type": "binary"},
    6: {"name": "BUTTON_BACK", "type": "binary"},
    7: {"name": "BUTTON_START", "type": "binary"},
    8: {"name": "BUTTON_LS", "type": "binary"},
    9: {"name": "BUTTON_RS", "type": "binary"},
    10: {"name": "DPAD_UP", "type": "binary"},
    11: {"name": "DPAD_DOWN", "type": "binary"},
    12: {"name": "DPAD_LEFT", "type": "binary"},
    13: {"name": "DPAD_RIGHT", "type": "binary"}
}

class JoystickController:
    def __init__(self, device_path="/dev/input/event10"):
        self.device_path = device_path
        self.device = None
        self.axis_values = defaultdict(float)
        self.button_values = defaultdict(int)
        self.running = False
        self.thread = None
        self.joystick_pub = rospy.Publisher("/joystick_state", JoystickState, queue_size=100)
        
    def connect(self):
        try:
            self.device = open(self.device_path, 'rb')
            rospy.loginfo(f"成功连接到手柄设备: {self.device_path}")
            return True
        except Exception as e:
            rospy.logerr(f"无法连接到手柄设备: {e}")
            return False
            
    def start(self):
        if not self.connect():
            return False
            
        self.running = True
        self.thread = threading.Thread(target=self._read_loop)
        self.thread.daemon = True
        self.thread.start()
        rospy.loginfo("手柄数据读取线程已启动")
        return True
        
    def stop(self):
        self.running = False
        if self.device:
            self.device.close()
        if self.thread:
            self.thread.join(timeout=1.0)
        rospy.loginfo("手柄数据读取线程已停止")
        
    def _read_loop(self):
        while self.running:
            try:
                # 读取完整的24字节事件数据
                data = self.device.read(24)
                if not data:
                    time.sleep(0.01)
                    continue
                    
                # 使用正确的格式字符串解包数据
                tv_sec, tv_usec, event_type, event_code, value = struct.unpack('llHHi', data)
                self._process_event(event_type, event_code, value)
                self._update_global_data()  # 更新全局变量
            except Exception as e:
                rospy.logerr(f"读取手柄数据时出错: {e}")
                time.sleep(0.5)
                self.device.close()
                if self.running:
                    self.connect()  

                    
    def _process_event(self, event_type, event_code, value):
        # 打印原始事件数据
        # rospy.loginfo(f"原始事件: type={event_type}, code={event_code}, value={value}")
    

        # 处理轴事件 (EV_ABS)
        if event_type == 3 and event_code in AXIS_CONFIG:
            config = AXIS_CONFIG[event_code]
            normalized = self._normalize_axis(value, config["min"], config["max"])
            self.axis_values[event_code] = self._apply_deadzone(normalized, config["deadzone"])
            if event_code == 9 or event_code == 10:
                self.axis_values[event_code] += 1
            rospy.logdebug(f"轴 {config['name']}: {self.axis_values[event_code]}")
            
        # 处理按钮事件 (EV_KEY)
        elif event_type == 1 and event_code in BUTTON_CONFIG:
            self.button_values[event_code] = value
            config = BUTTON_CONFIG[event_code]
            status = "按下" if value else "释放"
            rospy.logdebug(f"按钮 {config['name']}: {status}")
            
    def _normalize_axis(self, value, min_val, max_val):
        range_val = max_val - min_val
        normalized = (value - min_val) / range_val * 2 - 1
        return max(-1.0, min(1.0, normalized))
        
    def _apply_deadzone(self, value, deadzone):
        if abs(value) < deadzone:
            return 0.0
        return (value - deadzone * (value / abs(value))) / (1 - deadzone)
        
    def get_axis_value(self, axis_code):
        return self.axis_values.get(axis_code, 0.0)
        
    def get_button_value(self, button_code):
        return self.button_values.get(button_code, 0)
        
    def is_button_pressed(self, button_code):
        return self.get_button_value(button_code) == 1
    
    def _update_global_data(self):
        """将当前手柄数据更新到全局变量"""
        global global_left_x, global_left_y, global_right_x, global_right_y
        global global_trigger_left, global_trigger_right
        global global_button_a, global_button_b, global_button_x, global_button_y
        global global_button_lb, global_button_rb, global_button_back, global_button_start
        global global_button_ls, global_button_rs, global_dpad_up, global_dpad_down
        global global_dpad_left, global_dpad_right
        
        # 创建并发布消息
        msg = JoystickState()
        msg.header.stamp = rospy.Time.now()
        
        with data_lock:
            # 更新轴数据
            global_left_x = self.get_axis_value(0)
            global_left_y = self.get_axis_value(1)
            global_right_x = self.get_axis_value(2)
            global_right_y = self.get_axis_value(5)
            global_trigger_left = self.get_axis_value(10)
            global_trigger_right = self.get_axis_value(9)
            
            # 更新按钮数据
            global_button_a = self.is_button_pressed(304)
            global_button_b = self.is_button_pressed(305)
            global_button_x = self.is_button_pressed(307)
            global_button_y = self.is_button_pressed(308)
            global_button_lb = self.is_button_pressed(310)
            global_button_rb = self.is_button_pressed(311)

            # 同时填充消息数据
            msg.left_x = global_left_x
            msg.left_y = global_left_y
            msg.right_x = global_right_x
            msg.right_y = global_right_y
            msg.trigger_left = global_trigger_left
            msg.trigger_right = global_trigger_right
            
            msg.button_a = global_button_a
            msg.button_b = global_button_b
            msg.button_x = global_button_x
            msg.button_y = global_button_y
            msg.button_lb = global_button_lb
            msg.button_rb = global_button_rb
        
        # 发布消息
        self.joystick_pub.publish(msg)

def topicCallback(msg):
    rospy.loginfo("收到的消息: %s", msg.data)


def call_joint_moveit_ctrl_arm(joint_states, max_velocity=0.5, max_acceleration=0.5):
    rospy.wait_for_service("joint_moveit_ctrl_arm")
    try:
        moveit_service = rospy.ServiceProxy("joint_moveit_ctrl_arm", JointMoveitCtrl)
        request = JointMoveitCtrlRequest()
        request.joint_states = joint_states
        request.gripper = 0.0
        request.max_velocity = max_velocity
        request.max_acceleration = max_acceleration

        response = moveit_service(request)
        if response.status:
            rospy.loginfo("Successfully executed joint_moveit_ctrl_arm")
        else:
            rospy.logwarn(f"Failed to execute joint_moveit_ctrl_arm, error code: {response.error_code}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {str(e)}")

def call_joint_moveit_ctrl_gripper(gripper_position, max_velocity=0.5, max_acceleration=0.5):
    rospy.wait_for_service("joint_moveit_ctrl_gripper")
    try:
        moveit_service = rospy.ServiceProxy("joint_moveit_ctrl_gripper", JointMoveitCtrl)
        request = JointMoveitCtrlRequest()
        request.joint_states = [0.0] * 6
        request.gripper = gripper_position
        request.max_velocity = max_velocity
        request.max_acceleration = max_acceleration

        response = moveit_service(request)
        if response.status:
            rospy.loginfo("Successfully executed joint_moveit_ctrl_gripper")
        else:
            rospy.logwarn(f"Failed to execute joint_moveit_ctrl_gripper, error code: {response.error_code}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {str(e)}")

def call_joint_moveit_ctrl_piper(joint_states, gripper_position, max_velocity=0.5, max_acceleration=0.5):
    rospy.wait_for_service("joint_moveit_ctrl_piper")
    try:
        moveit_service = rospy.ServiceProxy("joint_moveit_ctrl_piper", JointMoveitCtrl)
        request = JointMoveitCtrlRequest()
        request.joint_states = joint_states
        request.gripper = gripper_position
        request.max_velocity = max_velocity
        request.max_acceleration = max_acceleration

        response = moveit_service(request)
        if response.status:
            rospy.loginfo("Successfully executed joint_moveit_ctrl_piper")
        else:
            rospy.logwarn(f"Failed to execute joint_moveit_ctrl_piper, error code: {response.error_code}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {str(e)}")

def convert_endpose(endpose):
    if len(endpose) == 6:
        x, y, z, roll, pitch, yaw = endpose
        qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw)
        return [x, y, z, qx, qy, qz, qw]

    elif len(endpose) == 7:
        return endpose  # 直接返回四元数

    else:
        raise ValueError("Invalid endpose format! Must be 6 (Euler) or 7 (Quaternion) values.")

def call_joint_moveit_ctrl_endpose(endpose, max_velocity=0.5, max_acceleration=0.5):
    rospy.wait_for_service("joint_moveit_ctrl_endpose")
    try:
        moveit_service = rospy.ServiceProxy("joint_moveit_ctrl_endpose", JointMoveitCtrl)
        request = JointMoveitCtrlRequest()
        
        request.joint_states = [0.0] * 6  # 填充6个关节状态
        request.gripper = 0.0
        request.max_velocity = max_velocity
        request.max_acceleration = max_acceleration
        request.joint_endpose = convert_endpose(endpose)  # 自动转换

        response = moveit_service(request)
        if response.status:
            rospy.loginfo("Successfully executed joint_moveit_ctrl_endpose")
        else:
            rospy.logwarn(f"Failed to execute joint_moveit_ctrl_endpose, error code: {response.error_code}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {str(e)}")

# 此处关节限制仅为测试使用，实际关节限制以READEME中为准
def randomval():
    arm_position = [
        random.uniform(-0.2, 0.2),  # 关节1
        random.uniform(0, 0.5),  # 关节2
        random.uniform(-0.5, 0),  # 关节3
        random.uniform(-0.2, 0.2),  # 关节4
        random.uniform(-0.2, 0.2),  # 关节5
        random.uniform(-0.2, 0.2)   # 关节6
    ]
    gripper_position = random.uniform(0, 0.035)

    return arm_position, gripper_position


def set_initial_position(service):
    """设置机械臂初始位置"""
    try:
        # 创建请求
        req = JointMoveitCtrlRequest()
        req.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        req.target_positions = [0.0, -1.57, 1.57, 0.0, 1.57, 0.0]  # 示例位置
        req.velocity = 0.2
        
        # 调用服务
        response = service(req)
        if response.success:
            rospy.loginfo("机械臂已移动到初始位置")
        else:
            rospy.logerr(f"设置初始位置失败: {response.message}")
    except Exception as e:
        rospy.logerr(f"调用服务失败: {e}")


def move_to_initial_pose():
    """移动到初始姿态（使用全局欧拉角）"""
    global global_endpose_euler
    rospy.loginfo("移动到初始末端姿态...")
    print(f"global_endpose_euler = [{global_endpose_euler[0]:.6f}, {global_endpose_euler[1]:.6f}, "
                  f"{global_endpose_euler[2]:.6f}, {global_endpose_euler[3]:.6f}, "
                  f"{global_endpose_euler[4]:.6f}, {global_endpose_euler[5]:.6f}]")
    call_joint_moveit_ctrl_endpose(global_endpose_euler)
    time.sleep(4.0)  # 等待机械臂到达目标姿态
    rospy.loginfo("机械臂已移动到初始位置")     

def control_robot_arm():
    """机械臂控制线程函数"""
    global global_arm_running, global_endpose_euler
    
    rospy.loginfo("机械臂控制线程已启动")
    
    

    # 控制参数（可根据需要调整）
    linear_speed = 0.001    # 线性移动速度 (m/次)
    angular_speed = 0.001   # 角度调整速度 (rad/次)
    deadzone = 0.4         # 死区阈值
    
    # 控制循环
    rate = rospy.Rate(20)  # 20Hz
    
    while not rospy.is_shutdown():
        # 读取手柄状态
        with data_lock:
            left_x = global_left_x      # 左摇杆X轴
            left_y = global_left_y      # 左摇杆Y轴
            right_x = global_right_x    # 右摇杆X轴
            right_y = global_right_y    # 右摇杆Y轴
            button_lb = global_button_lb  # 左肩部按钮
            button_rb = global_button_rb  # 右肩部按钮
            button_a = global_button_a    # 按钮A
            button_b = global_button_b    # 按钮B
            button_x = global_button_x    # 按钮X
            button_y = global_button_y    # 按钮Y 
        
        # 1. 左摇杆控制末端位置(X, Y)
        if abs(left_x) > deadzone or abs(left_y) > deadzone:
            global_endpose_euler[0] += -left_y * linear_speed  # Y轴控制X坐标
            global_endpose_euler[1] += -left_x * linear_speed # X轴控制Y坐标
        
        # 2. 右摇杆X轴控制末端roll角
        if abs(right_x) > deadzone:
            global_endpose_euler[3] += right_x * angular_speed  # roll角
        
        # 3. 右摇杆Y轴控制末端pitch角
        if abs(right_y) > deadzone:
            global_endpose_euler[4] += -right_y * angular_speed  # pitch角
        
        # 4. 左右肩部按钮控制末端Z坐标
        if button_lb:  # 左肩部按钮：上升
            global_endpose_euler[2] += linear_speed
        if button_rb:  # 右肩部按钮：下降
            global_endpose_euler[2] -= linear_speed
        
        # 5. 按钮A/B控制yaw角（偏航角）
        if button_a:  # 按钮A：增加yaw角
            global_endpose_euler[5] += angular_speed
        if button_b:  # 按钮B：减少yaw角
            global_endpose_euler[5] -= angular_speed
        
        
        print(f"global_endpose_euler = [{global_endpose_euler[0]:.6f}, {global_endpose_euler[1]:.6f}, "
            f"{global_endpose_euler[2]:.6f}, {global_endpose_euler[3]:.6f}, "
            f"{global_endpose_euler[4]:.6f}, {global_endpose_euler[5]:.6f}]")
        
        rate.sleep()  # 属于while循环

    rospy.loginfo("机械臂控制线程已停止")

def main():
    rospy.init_node("test_joint_moveit_ctrl", anonymous=True)
    # rospy.init_node("subscriber")

    # 初始化手柄控制器
    joystick = JoystickController(device_path="/dev/input/event10")
    if not joystick.start():
        rospy.logfatal("无法启动手柄控制器，程序退出")
        return
        
    rospy.loginfo("手柄控制器已启动，开始读取手柄数据")

    move_to_initial_pose()

    # 启动机械臂控制线程
    global_arm_running = True
    global_arm_thread = threading.Thread(target=control_robot_arm)
    global_arm_thread.daemon = True
    global_arm_thread.start()
    rospy.loginfo("机械臂控制线程已启动")

     # 主循环
    rate = rospy.Rate(1)  # 1Hz
    
    try:
        while not rospy.is_shutdown():
            # 主循环可以保持简单，主要逻辑在其他线程中处理
            # 6. 调用API更新末端姿态
            # call_joint_moveit_ctrl_endpose(global_endpose_euler)
            rate.sleep()
    except KeyboardInterrupt:
        rospy.loginfo("程序被用户中断")
    finally:
        # 程序退出时清理资源
        global_arm_running = False
        if global_arm_thread:
            global_arm_thread.join(timeout=2.0)
            
        joystick.stop()
        rospy.loginfo("程序已安全退出")


if __name__ == "__main__":
    main()

    # # 5.设置循环调用回调函数
    # # rospy.spin()
    # arm_position, gripper_position = [], 0

    # try:
    #     rate = rospy.Rate(1000)  # 20Hz读取频率
    #     last_print_time = time.time()
        
    #     while not rospy.is_shutdown():
    #         # 1. 读取所有轴数据
    #         axis_data = {
    #             0: joystick.get_axis_value(0),  # 左摇杆X轴
    #             1: joystick.get_axis_value(1),  # 左摇杆Y轴
    #             2: joystick.get_axis_value(2),  # 右摇杆X轴
    #             5: joystick.get_axis_value(5),  # 右摇杆Y轴
    #             10: joystick.get_axis_value(10),  # 左扳机
    #             9: joystick.get_axis_value(9)   # 右扳机
    #         }
            
    #         # 2. 读取所有按钮数据
    #         button_data = {
    #             304: joystick.is_button_pressed(304),  # 按钮A
    #             305: joystick.is_button_pressed(305),  # 按钮B
    #             307: joystick.is_button_pressed(307),  # 按钮X
    #             308: joystick.is_button_pressed(308),  # 按钮Y
    #             310: joystick.is_button_pressed(310),  # 左 bumper
    #             311: joystick.is_button_pressed(311),  # 右 bumper
    #         }
            
    #         # 3. 每0.5秒打印一次完整数据（避免刷屏）
    #         current_time = time.time()
    #         if current_time - last_print_time >= 0.1:
    #             # rospy.loginfo("=== 手柄实时数据 ===")
    #             rospy.loginfo(f"轴数据: {axis_data}")
    #             rospy.loginfo(f"按钮数据: {button_data}")
    #             last_print_time = current_time
        
    #         # 4. 控制循环频率
    #         rate.sleep()
            
    # except rospy.ROSInterruptException:
    #     pass
    # finally:
    #     joystick.stop()
    #     rospy.loginfo("手柄数据读取器已退出")


    #for i in range(10): 
        # arm_position, _ = randomval()  # 机械臂控制
        # call_joint_moveit_ctrl_arm(arm_position, max_velocity=0.5, max_acceleration=0.5)
        # time.sleep(1)
        # _, gripper_position = randomval()  # 夹爪控制
        # call_joint_moveit_ctrl_gripper(gripper_position)
        # time.sleep(1)
        # arm_position, gripper_position = randomval()
        # call_joint_moveit_ctrl_piper(arm_position, gripper_position)  # 机械臂夹爪联合控制
        # time.sleep(1)

        #endpose_euler = [0.531014, -0.133376, 0.418909, -0.6052452780065936, 1.2265301318390152, -0.9107128036906411]
        #endpose_euler = [0.2531014, 0, 0.218909, -0.6052452780065936, 1.2265301318390152, -0.9107128036906411]
        # call_joint_moveit_ctrl_endpose(endpose_euler)  # 末端位置控制(欧拉角)
        #time.sleep(1)

        # endpose_quaternion = [0.531014, -0.133376, 0.418909, 0.02272779901175584, 0.6005891177332143, -0.18925185045722595, 0.7765049233012219]
        # call_joint_moveit_ctrl_endpose(endpose_quaternion)  # 末端位置控制(四元数)
        # time.sleep(1)
        # arm_position = [0, 0, 0, 0, 0, 0]
        
        # call_joint_moveit_ctrl_arm(arm_position, max_velocity=0.5, max_acceleration=0.5) # 回零
        # time.sleep(1)