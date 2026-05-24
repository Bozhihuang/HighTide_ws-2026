# 6-DOF normalized thruster command (-1.0 to 1.0)
std_msgs/Header header
float32 surge       # Forward/backward  (positive = forward)
float32 sway        # Left/right        (positive = right)
float32 heave       # Up/down           (positive = up, only used when NOT in depth setpoint mode)
float32 yaw         # Rotation          (positive = clockwise)
float32 pitch       # Nose up/down      (positive = nose up, usually 0 in Alt Hold)
float32 roll        # Roll              (positive = right roll, usually 0 in Alt Hold)
