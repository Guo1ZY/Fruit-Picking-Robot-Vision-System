需要注意功能包 [moveit_ctrl](moveit_ctrl) 和 [xbox_chassis](xbox_chassis) 都需要放在piper_ros/src/piper_moveit/下进行编译，由于需要msg的支持

当此系统放在arm架构下时，apple_detection的python调用会出一些问题，pytorch调用导入不起的问题，似乎和多线程调用相关，@Guo1ZY 正在逐步解决当中
