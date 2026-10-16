#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "traversable_node/traversable_node.hpp"

#include "nlohmann/json.hpp"
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/image_encodings.hpp"

namespace traversable_node
{
  TraversableNode::TraversableNode(const rclcpp::NodeOptions &options)
  : Node("traversable_node", options),
    camera_k_(Eigen::Matrix3d::Zero())
  {
    std::string inference_topic_name = declare_parameter<std::string>("inference_topic", "/inference");

    bool camera_info_via_json = declare_parameter<bool>("camera_info_via_json", true);
    std::string camera_info_topic_name;
    std::filesystem::path camera_info_json_path;
    
    if (camera_info_via_json)
    {
      const char* home = std::getenv("HOME");
      if (!home)
        throw std::runtime_error("Переменная HOME не задана");

      std::string camera_json_path_str = declare_parameter<std::string>("camera_info_json_path", ".ros/camera_info.json");
      
      camera_info_json_path = std::filesystem::path(home) / std::filesystem::path(camera_json_path_str);
    }
    else
      camera_info_topic_name = declare_parameter<std::string>("camera_info_topic", "/camera_info");
      
    std::string vis_topic_name = declare_parameter<std::string>("visual_topic", "/visualization_marker");

    lookup_timeout_ = declare_parameter<double>("tf_lookup_timeout", 2.0);
    max_lookup_range_ = declare_parameter<double>("max_lookup_range", 10.0); 

    camera_frame_ = declare_parameter<std::string>("camera_frame", "camera_optical_frame");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    footprint_frame_ = declare_parameter<std::string>("footprint_frame", "base_footprint");

    buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    listener_ = std::make_unique<tf2_ros::TransformListener>(*buffer_);

    inference_sub_ = create_subscription<sensor_msgs::msg::Image>(inference_topic_name, 1, std::bind(&TraversableNode::inferenceCallback, this, std::placeholders::_1));
    if (camera_info_via_json)
    {
      std::ifstream file(camera_info_json_path);
      if (!file)
      {
        std::string exception = "TraversableNode::TraversableNode: can't open file " + camera_info_json_path.string();
        throw std::runtime_error(exception);
      }

      nlohmann::json data;
      file >> data;

      const auto& rows = data.at("K");
      if (!rows.is_array() || rows.size() != 3)
        throw std::runtime_error("TraversableNode::TraversableNode: K matrix should have 3 x 3 dimension");

      for (size_t idx = 0; idx < 3; ++idx)
      {
        if (!rows[idx].is_array() || rows[idx].size() != 3)
          throw std::runtime_error("TraversableNode::TraversableNode: K matrix should have 3 x 3 dimension");

        for (int jdx = 0; jdx < 3; ++jdx)
          camera_k_(idx, jdx) = rows[idx][jdx].get<double>();
      }

      RCLCPP_INFO(get_logger(), "fx = %.2f fy = %.2f cx = %.2f cy = %.2f",
                                camera_k_(0, 0), camera_k_(1, 1), camera_k_(0, 2), camera_k_(1, 2));
    }
    else
      cam_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(camera_info_topic_name, 1, std::bind(&TraversableNode::cameraInfoCallback, this, std::placeholders::_1));

    vis_pub_ = create_publisher<visualization_msgs::msg::Marker>(vis_topic_name, 1);

    RCLCPP_INFO(get_logger(), "TraversableNode::TraversableNode: start");
  }

  void TraversableNode::inferenceCallback(sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    if (camera_k_.isZero())
    {
      RCLCPP_WARN(get_logger(), "TraversableNode::inferenceCallback: didn't have K instricincts matrix of camera! Can't project points! Retry later!");
      return;
    }

    cv::Mat mask;

    try {
      mask = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8)->image;
    }
    catch (const cv_bridge::Exception& e) {
      RCLCPP_ERROR(get_logger(), "Image conversion failed: %s", e.what());
      return;
    }

    vector_int_pair borders = scanlineDetector(mask);

    std::optional<Eigen::Isometry3d> base_to_camera = getTransform(camera_frame_, base_frame_);
    std::optional<Eigen::Isometry3d> ground_to_base = getTransform(footprint_frame_, base_frame_);

    if (!base_to_camera || !ground_to_base)
    {
      RCLCPP_WARN(get_logger(), "TraversableNode::inferenceCallback: can't found needed transformation! Retry later!");
      return;
    }

    Eigen::Matrix3d hom_ground_image = groundToImageHomography(camera_k_, *base_to_camera, *ground_to_base);
    Eigen::Matrix3d hom_image_ground = hom_ground_image.inverse();

    std::vector<size_t> valid_y;
    size_t border_size = borders.size();
    valid_y.reserve(border_size);

    for (size_t y = 0; y < border_size; ++y)
    {
      const auto& [left, right] = borders[y];

      if (left == -1)
        continue;

      Eigen::Vector3d left_point = hom_image_ground * Eigen::Vector3d(left, y, 1.0);
      Eigen::Vector3d right_point = hom_image_ground * Eigen::Vector3d(right, y, 1.0);

      if (!left_point.allFinite() || !right_point.allFinite() ||
           left_point.z() <= 1e-9 ||  right_point.z() <= 1e-9)
        continue;
      
      Eigen::Vector3d left_point_norm(left_point.x() / left_point.z(), left_point.y() / left_point.z(), 0.0);
      Eigen::Vector3d right_point_norm(right_point.x() / right_point.z(), right_point.y() / right_point.z(), 0.0);

      if (!left_point_norm.allFinite() || !right_point_norm.allFinite() ||
           left_point_norm.head<2>().norm() > max_lookup_range_ || right_point_norm.head<2>().norm() > max_lookup_range_)
        continue;

      valid_y.push_back(y);
    }

    if (valid_y.size() < 2)
    {
      RCLCPP_WARN(get_logger(), "Not enough valid rows to fit borders");
      return;
    }

    Eigen::Index valid_size = static_cast<Eigen::Index>(valid_y.size());
    Eigen::Matrix<double, Eigen::Dynamic, 2> A_left(valid_size, 2);
    Eigen::Matrix<double, Eigen::Dynamic, 2> A_right(valid_size, 2);
    Eigen::VectorXd B_left(valid_size);
    Eigen::VectorXd B_right(valid_size);

    for (Eigen::Index idx = 0; idx < valid_size; ++idx)
    {
      size_t y = valid_y[idx];
      const auto& [left, right] = borders[y];

      A_left(idx, 0) = static_cast<double>(y);
      A_left(idx, 1) = 1.0;
      B_left(idx) = left;

      A_right(idx, 0) = static_cast<double>(y);
      A_right(idx, 1) = 1.0;
      B_right(idx) = right;
    }

    auto [left_border, right_border] = lineExtractor(A_left, A_right, B_left, B_right);

    auto [left_border_ground, right_border_ground] = projectImageLineToGround(left_border, right_border, hom_ground_image);
    auto area = projectAreaToGround(borders, hom_image_ground, max_lookup_range_);

    Line3d mid_line_ground;
    mid_line_ground(0) = (1.0 / 2.0) * ((left_border_ground(0) / left_border_ground(1)) + (right_border_ground(0) / right_border_ground(1)));
    mid_line_ground(1) = 1.0;
    mid_line_ground(2) = (1.0 / 2.0) * ((left_border_ground(2) / left_border_ground(1)) + (right_border_ground(2) / right_border_ground(1)));

    std::vector<Eigen::Vector3d> left_border_sample = sampleLine(left_border_ground);
    std::vector<Eigen::Vector3d> mid_line_sample = sampleLine(mid_line_ground);
    std::vector<Eigen::Vector3d> right_border_sample = sampleLine(right_border_ground);

    publishLines(left_border_sample, mid_line_sample, right_border_sample);
    publishArea(area);
  }

  void TraversableNode::cameraInfoCallback(sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)
  {
    camera_k_ << msg->k[0], msg->k[1], msg->k[2],
                msg->k[3], msg->k[4], msg->k[5],
                msg->k[6], msg->k[7], msg->k[8];

    RCLCPP_INFO(get_logger(), "fx = %.2f fy = %.2f cx = %.2f cy = %.2f",
                              camera_k_(0, 0), camera_k_(1, 1), camera_k_(0, 2), camera_k_(1, 2));
  }

  void TraversableNode::publishLines(const std::vector<Eigen::Vector3d>& left_border,
                                     const std::vector<Eigen::Vector3d>& mid_line,
                                     const std::vector<Eigen::Vector3d>& right_border)
  {
    visualization_msgs::msg::Marker line_marker;
    line_marker.header.frame_id = footprint_frame_;
    line_marker.header.stamp = now();
    line_marker.ns = "traverse_line";
    line_marker.id = 0;
    line_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    line_marker.action = visualization_msgs::msg::Marker::ADD;
    line_marker.pose.orientation.w = 1.0;
    line_marker.scale.x = 0.03;
    line_marker.color.r = 1.0f;
    line_marker.color.a = 1.0f;

    for(const auto& point : left_border)
    {
      geometry_msgs::msg::Point p;
      p.x = point(0);
      p.y = point(1);
      p.z = point(2);
      line_marker.points.push_back(p);
    }

    vis_pub_->publish(line_marker);

    line_marker.id = 1;
    line_marker.color.r = 0.0f;
    line_marker.color.g = 1.0f;

    line_marker.points.clear();

    for(const auto& point : mid_line)
    {
      geometry_msgs::msg::Point p;
      p.x = point(0);
      p.y = point(1);
      p.z = point(2);
      line_marker.points.push_back(p);
    }

    vis_pub_->publish(line_marker);

    line_marker.id = 2;
    line_marker.color.r = 1.0f;
    line_marker.color.g = 0.0f;

    line_marker.points.clear();

    for(const auto& point : right_border)
    {
      geometry_msgs::msg::Point p;
      p.x = point(0);
      p.y = point(1);
      p.z = point(2);
      line_marker.points.push_back(p);
    }

    vis_pub_->publish(line_marker);
  }

  void TraversableNode::publishArea(const std::vector<Eigen::Vector3d>& area)
  {
    visualization_msgs::msg::Marker area_marker;
    area_marker.header.frame_id = footprint_frame_;
    area_marker.header.stamp = now();
    area_marker.ns = "area";
    area_marker.id = 0;
    area_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    area_marker.action = visualization_msgs::msg::Marker::ADD;
    area_marker.pose.orientation.w = 1.0;
    area_marker.scale.x = 0.03;
    area_marker.color.r = 1.0f;
    area_marker.color.a = 1.0f;

    for(auto&& point : area)
    {
      geometry_msgs::msg::Point p;
      p.x = point(0);
      p.y = point(1);
      p.z = point(2);
      area_marker.points.push_back(p);
    }

    vis_pub_->publish(area_marker);
  }

  std::optional<Eigen::Isometry3d> TraversableNode::getTransform(const std::string& frame, const std::string& child_frame)
  {
    try {
      auto tf_transform = buffer_->lookupTransform(child_frame, frame, now(), rclcpp::Duration::from_seconds(lookup_timeout_));

      Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
      transform.translation() = Eigen::Vector3d(tf_transform.transform.translation.x,
                                                tf_transform.transform.translation.y,
                                                tf_transform.transform.translation.z);
      Eigen::Quaterniond q(tf_transform.transform.rotation.w,
                           tf_transform.transform.rotation.x,
                           tf_transform.transform.rotation.y,
                           tf_transform.transform.rotation.z);
      transform.linear() = q.normalized().toRotationMatrix();
      return transform;
    } catch (const tf2::TransformException& ex) {
      RCLCPP_WARN(get_logger(), "Waiting for TF %s -> %s: %s", child_frame.c_str(), frame.c_str(), ex.what());
      return std::nullopt;
    }
  }

  TraversableNode::vector_int_pair TraversableNode::scanlineDetector(cv::Mat img)
  {
    std::vector<std::pair<int, int>> result(img.rows);

    int img_rows = static_cast<int>(img.rows);
    int img_cols = static_cast<int>(img.cols);

    for (int y = 0; y < img_rows; ++y)
    {
      uint8_t *row = img.ptr<uint8_t>(y);

      int right_idx = 0;
      int left_idx = right_idx;

      int max_gap = 0;
      int gap_right_idx = -1;
      int gap_left_idx = -1;

      while (right_idx < img_cols)
      {
        if (row[right_idx] == 0)
        {
          if (right_idx - left_idx > max_gap)
          {
            max_gap = right_idx - left_idx;
            gap_left_idx = left_idx;
            gap_right_idx = right_idx;
          }

          right_idx++;
          left_idx = right_idx;
        }
        else
          right_idx++;
      }

      if (right_idx - left_idx > max_gap)
      {
        gap_left_idx = left_idx;
        gap_right_idx = right_idx;
      }

      result[y] = {gap_left_idx, gap_right_idx};
    }

    return result;
  }

  std::tuple<TraversableNode::Line, TraversableNode::Line> TraversableNode::lineExtractor(const Eigen::Matrix<double, Eigen::Dynamic, 2> &A_left,
                                                       const Eigen::Matrix<double, Eigen::Dynamic, 2> &A_right,
                                                       const Eigen::VectorXd &B_left,
                                                       const Eigen::VectorXd &B_right)
  {
    Line left_border = A_left.colPivHouseholderQr().solve(B_left);
    Line right_border = A_right.colPivHouseholderQr().solve(B_right);

    return {left_border, right_border};
  }

  Eigen::Matrix3d TraversableNode::groundToImageHomography(const Eigen::Matrix3d& K,
                                                           const Eigen::Isometry3d& base_to_camera,
                                                           const Eigen::Isometry3d& footprint_to_base)
  {
    Eigen::Isometry3d camera_to_base = base_to_camera.inverse();
    const Eigen::Matrix3d& rot = camera_to_base.linear();
    const Eigen::Vector3d& trans = camera_to_base.translation();

    Eigen::Matrix3d H;
    H.col(0) = rot.col(0);
    H.col(1) = rot.col(1);
    H.col(2) = trans + footprint_to_base.translation()(2) * rot.col(2);

    return K * H;
  }

  std::tuple<TraversableNode::Line3d, TraversableNode::Line3d> TraversableNode::projectImageLineToGround(Line left_border, Line right_border, const Eigen::Matrix3d& hom_ground_image)
  {
    Line3d left_border_3d;
    left_border_3d << 1.0, -left_border(0), -left_border(1);
    Line3d right_border_3d;
    right_border_3d << 1.0, -right_border(0), -right_border(1);

    Line3d left_border_ground = hom_ground_image.transpose() * left_border_3d;
    Line3d right_border_ground = hom_ground_image.transpose() * right_border_3d;

    double left_norm = left_border_ground.head<2>().norm();
    double right_norm = right_border_ground.head<2>().norm();

    if (left_norm < 1e-9 || right_norm < 1e-9)
      throw std::runtime_error("TraversableNode::projectImageLineToGround: degenerate ground line!");

    return {left_border_ground, right_border_ground};
  }

  std::vector<Eigen::Vector3d> TraversableNode::projectAreaToGround(const vector_int_pair& area, const Eigen::Matrix3d& hom_image_ground, double max_lookup_range)
  {
    std::vector<Eigen::Vector3d> result;
    result.reserve(2 * area.size());

    size_t area_size = area.size();
    for (size_t y = 0; y < area_size; ++y)
    {
      if (area[y].first < 0 || area[y].second < 0)
        continue;

      Eigen::Vector3d p_left(area[y].first, y, 1.0);
      Eigen::Vector3d p_right(area[y].second, y, 1.0);
      Eigen::Vector3d ground_left = hom_image_ground * p_left;
      Eigen::Vector3d ground_right = hom_image_ground * p_right;

      if (!ground_left.allFinite() || std::abs(ground_left.z()) < 1e-9)
        continue;

      Eigen::Vector3d ground_left_norm(ground_left.x() / ground_left.z(), ground_left.y() / ground_left.z(), 0.0);

      if (!ground_left_norm.allFinite() || ground_left_norm.head<2>().norm() > max_lookup_range)
        continue;

      result.push_back(ground_left_norm);

      if (!ground_right.allFinite() || std::abs(ground_right.z()) < 1e-9)
        continue;

      Eigen::Vector3d ground_right_norm(ground_right.x() / ground_right.z(), ground_right.y() / ground_right.z(), 0.0);

      if (!ground_right_norm.allFinite() || ground_right_norm.head<2>().norm() > max_lookup_range)
        continue;

      result.push_back(ground_right_norm);
    }

    return result;
  }

  std::vector<Eigen::Vector3d> TraversableNode::sampleLine(const Line3d& line, double depth, double step)
  {
    size_t sample_count = static_cast<size_t>(depth / step);
    std::vector<Eigen::Vector3d> sample_set;
    sample_set.reserve(sample_count);

    const double A = line(0);
    const double B = line(1);
    const double C = line(2);

    if (std::abs(B) < 1e-9)
      throw std::runtime_error("TraversableNode::sampleLine: cannot represent line as Y(X)");

    for (double x = 0.0; x <= depth; x += step)
    {
      double y = -(A * x + C) / B;

      sample_set.push_back(Eigen::Vector3d(x, y, 0.0));
    }

    return sample_set;
  }
} // namespace traversable_node
