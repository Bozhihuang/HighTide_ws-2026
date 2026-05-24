# Current navigation tier and tracking quality
uint8 VSLAM=1
uint8 VIO=2
uint8 DEAD_RECKONING=3

uint8 current_tier
float32 tracking_confidence     # 0.0 - 1.0
string status_message
