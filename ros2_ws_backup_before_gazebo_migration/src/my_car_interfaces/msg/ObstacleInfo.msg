# ObstacleInfo.msg
# Standardized message for YOLO AI Detection output

string object_label      # "Stone", "Child", "Vehicle", etc.
float32 distance         # Estimated distance in meters
float32 height           # Estimated height in meters (to check if we can drive over it)
bool is_passable         # True if Stone/Low, False if Child/Big Object