cmake_minimum_required(VERSION 3.8)
project(my_car_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# 1. Install the Launch Files (The Ignition Keys)
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# 2. Install the URDF (The Body)
install(DIRECTORY
  urdf
  DESTINATION share/${PROJECT_NAME}
)

# 3. Install RViz Configs (The Visualizer Settings)
install(DIRECTORY
  rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()