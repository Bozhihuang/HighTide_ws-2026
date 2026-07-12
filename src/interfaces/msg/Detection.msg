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

# Instance-segmentation extras (populated only by *-seg YOLO models; empty/0.0
# for plain detect models). center_x/center_y are the MASK centroid when a mask
# is present, which tracks better than the bbox center for thin/rotated/occluded
# shapes (e.g. slalom poles). mask_polygon is a flattened [x0,y0,x1,y1,...]
# contour in image pixel coords.
float32 mask_area              # segmented area in pixels (0.0 if no mask)
float32[] mask_polygon         # flattened contour points, image pixel coords
