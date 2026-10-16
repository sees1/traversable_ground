#pragma once

#include <vector>
#include <string>
#include <tuple>
#include <optional>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "visualization_msgs/msg/marker.hpp"

#include "tf2_ros/buffer.hpp"
#include "tf2_ros/transform_listener.hpp"

namespace traversable_node
{

  class TraversableNode final : public rclcpp::Node
  {
  public:
    using vector_int_pair = std::vector<std::pair<int, int>>;
    using Line = Eigen::Vector2d;
    using Line3d = Eigen::Vector3d;

  public:
    explicit TraversableNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

  private:
    void inferenceCallback(sensor_msgs::msg::Image::ConstSharedPtr msg);
    void cameraInfoCallback(sensor_msgs::msg::CameraInfo::ConstSharedPtr msg);
    void publishLines(const std::vector<Eigen::Vector3d>& left_border,
                      const std::vector<Eigen::Vector3d>& mid_line,
                      const std::vector<Eigen::Vector3d>& right_border);
    void publishArea(const std::vector<Eigen::Vector3d>& area);

    std::optional<Eigen::Isometry3d> getTransform(const std::string& frame, const std::string& child_frame);

  private:
    static vector_int_pair scanlineDetector(cv::Mat img);
    static std::tuple<Line, Line> lineExtractor(const Eigen::Matrix<double, Eigen::Dynamic, 2>& A_left,
                                                const Eigen::Matrix<double, Eigen::Dynamic, 2>& A_right,
                                                const Eigen::VectorXd& B_left,
                                                const Eigen::VectorXd& B_right);
    static Eigen::Matrix3d groundToImageHomography(const Eigen::Matrix3d& K,
                                                   const Eigen::Isometry3d& base_to_camera,
                                                   const Eigen::Isometry3d& footprint_to_base);
    static std::tuple<Line3d, Line3d> projectImageLineToGround(Line left_border, 
                                                               Line right_border,
                                                               const Eigen::Matrix3d& hom_ground_image);
    static std::vector<Eigen::Vector3d> projectAreaToGround(const vector_int_pair& area,
                                                            const Eigen::Matrix3d& hom_image_ground,
                                                            double max_lookup_range);
    static std::vector<Eigen::Vector3d> sampleLine(const Line3d& line, double depth = 5.0, double step = 0.1);

  private:
    Eigen::Matrix3d camera_k_;

    std::unique_ptr<tf2_ros::Buffer> buffer_;
    std::unique_ptr<tf2_ros::TransformListener> listener_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr inference_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr vis_pub_;

    double lookup_timeout_;
    double max_lookup_range_;
    std::string base_frame_;
    std::string camera_frame_;
    std::string footprint_frame_;
  };

} // namespace traversable_node