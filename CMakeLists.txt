cmake_minimum_required(VERSION 3.10)
project(traversable_corridor)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)

set(rosdep rclcpp
           cv_bridge
           sensor_msgs
           visualization_msgs
           tf2_ros
           tf2)

add_executable(traversable_node src/main.cpp 
                                src/traversable_node.cpp)
target_compile_features(traversable_node PUBLIC cxx_std_17)
target_compile_options(traversable_node PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(traversable_node PUBLIC include
                                                   ${OpenCV_INCLUDE_DIRS})
target_link_libraries(traversable_node PUBLIC ${OpenCV_LIBS}
                                              Eigen3::Eigen
                                              nlohmann_json::nlohmann_json)
ament_target_dependencies(traversable_node PUBLIC ${rosdep})


install(TARGETS traversable_node
        DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/segment.py
                 scripts/inference_ipc.py
                 scripts/image_socket_bridge.py
                 scripts/mp4_publisher.py
        DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch runs video
        DESTINATION share/${PROJECT_NAME})

ament_package()
