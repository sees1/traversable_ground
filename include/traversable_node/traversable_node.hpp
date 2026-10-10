#pragma once

#include <vector>
#include <string>
#include <tuple>
#include <optional>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>
#include "sensor_msgs/msg/image.hpp"

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

    void publishMarkers(const Line3d& left_border, const Line3d& right_border);

    std::optional<Eigen::Isometry3d> getTransform(const std::string& frame, const std::string& child_frame);

  private:
    static vector_int_pair scanlineDetection(cv::Mat img);
    static std::tuple<Line, Line> lineExtractor(const Eigen::Matrix<double, Eigen::Dynamic, 2>& A_left,
                                                const Eigen::Matrix<double, Eigen::Dynamic, 2>& A_right,
                                                const Eigen::VectorXd& B_left,
                                                const Eigen::VectorXd& B_right);
    static Eigen::Matrix3d groundToImageHomography(const Matrix3d& K,
                                                   const Eigen::Isometry3d& base_to_camera,
                                                   const Eigen::Isometry3d& footprint_to_base);
    static std::tuple<Line3d, Line3d> projectImageLineToGround(Line left_border, 
                                                               Line right_border,
                                                               const Eigen::Matrix3d& hom_ground_image);

  private:
    Eigen::Matrix3d camera_k_;

    double lookup_timeout_;
    std::string base_frame_;
    std::string camera_frame_;
    std::string footprint_frame_;
  };

} // namespace traversable_node