cmake_minimum_required(VERSION 3.10)
project(traversable_node)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(traversable_node src/main.cpp 
                                src/traversable_node.cpp)
target_compile_features(traversable_node PUBLIC cxx_std_17)
target_compile_options(traversable_node PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(traversable_node PRIVATE include)
target_link_directories(traversable_node PRIVATE ${OpenCV_INCLUDE_DIRS}
                                                 Eigen3::Eigen)
ament_target_dependencies(traversable_node PUBLIC
  rclcpp
  cv_bridge)


install(TARGETS traversable_node
        DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch
        DESTINATION share/${PROJECT_NAME})

ament_package()