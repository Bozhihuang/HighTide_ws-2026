# Single YOLO detection result
std_msgs/Header header
int32 class_id
string class_name
float32 confidence
float32 x_min
float32 y_min
float32 x_max
float32 y_max
float32 center_x
float32 center_y
float32 depth_m        # Depth from ZED stereo, -1.0 if unavailable
float32 width
float32 height
