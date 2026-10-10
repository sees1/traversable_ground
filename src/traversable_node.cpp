#include "traversable_node/traversable_node.hpp"

#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/image_encodings.hpp"

namespace traversable_node
{
  TraversableNode::TraversableNode(const rclcpp::NodeOptions &options)
  : Node("traversable_node", options),
    camera_k_(Eigen::Matrix3d::Zero())
  {
    lookup_timeout_ = declare_parameter<double>("tf_lookup_timeout", 2.0);
    camera_frame_ = declare_parameter<std::string>("camera_frame", "camera_optical_frame");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    footprint_frame_ = declare_parameter<std::string>("footprint_frame", "base_footprint");

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
    }

    vector_int_pair borders = scanlineDetector(mask);

    Eigen::Matrix<double, Eigen::Dynamic, 2> A_left(static_cast<Eigen::Index>(borders.size()), 2);
    Eigen::Matrix<double, Eigen::Dynamic, 2> A_right(static_cast<Eigen::Index>(borders.size()), 2);
    Eigen::VectorXd B_left(static_cast<Eigen::Index>(borders.size()));
    Eigen::VectorXd B_right(static_cast<Eigen::Index>(borders.size()));

    Eigen::Index rows = A_left.rows();

    for (Eigen::Index idx = 0; idx < rows; ++idx)
    {
      A_left(idx, 0) = borders[idx].first;
      A_left(idx, 1) = 1;
      B_left(idx) = idx;

      A_right(idx, 0) = boarders[idx].second;
      A_right(idx, 1) = 1;
      B_right(idx) = idx;
    }

    Line left_border, right_border, middle_line;
    auto [left_border, middle_line, right_border] = lineExtractor(A_left, A_right, B_left, B_right);
    
    std::optional<Eigen::Isometry3d> base_to_camera = getTransform(camera_frame_, base_frame_);
    std::optional<Eigen::Isometry3d> ground_to_base = getTransform(footprint_frame_, base_frame_);

    if (!base_to_camera || !ground_to_base)
    {
      RCLCPP_WARN(get_logger(), "TraversableNode::inferenceCallback: can't found needed transformation! Retry later!");
      return;
    }
  
    Eigen::Matrix3d hom_ground_image = groundToImageHomography(camera_k_, *base_to_camera, *ground_to_base);

    Line3d left_border_ground, right_border_ground;
    auto [left_border_ground, right_border_ground] = projectImageLineToGround(left_border, right_border, hom_ground_image);

    double a_l = -1.0 * (left_border_ground(0) / left_border_ground(1));
    double b_l = -1.0 * (left_border_ground(2) / left_border_ground(1));
    double a_l = -1.0 * (right_border_ground(0) / right_border_ground(1));
    double b_l = -1.0 * (right_border_ground(2) / right_border_ground(1));


    publishMarkers();
  }

  void TraversableNode::cameraInfoCallback(sensor_msgs::msg::CameraInfo::ConstSharedPtr msg)
  {
    camera_k << msg->k[0], msg->k[1], msg->k[2],
                msg->k[3], msg->k[4], msg->k[5],
                msg->k[6], msg->k[7], msg->k[8];

    RCLCPP_INFO(get_logger(), "fx = %.2f fy = %.2f cx = %.2f cy = %.2f",
                              camera_k(0, 0), camera(1, 1), camera(0, 2), camera(1, 2));
  }

  std::optional<Eigen::Isometry3d> TraversableNode::getTransform(const std::string& frame, const std::string& child_frame)
  {
    try {
      auto tf_transform = buffer_->lookupTransform(child_frame, frame, now(), rclcpp::Duration::from_seconds(lookup_timeout_))

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

  std::tuple<Line Line> TraversableNode::lineExtractor(const Eigen::Matrix<double, Eigen::Dynamic, 2> &A_left,
                                                       const Eigen::Matrix<double, Eigen::Dynamic, 2> &A_right,
                                                       const Eigen::VectorXd &B_left,
                                                       const Eigen::VectorXd &B_right)
  {
    Line left_border = A_left.colPivHousehoderQr().solve(B_left);
    Line right_border = A_right.colPivHouseholderQr().solve(B_right);

    return {left_border, right_border};
  }

  Eigen::Matrix3d TraversableNode::groundToImageHomography(const Matrix3d& K,
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

  std::tuple<Line3d, Line3d> TraversableNode::projectImageLineToGround(Line left_border, Line right_border, const Eigen::Matrix3d& hom_ground_image)
  {
    Line3d left_border_3d;
    left_broder_3d << left_border(0), -1.0, left_border(1);
    Line3d right_border_3d;
    right_border_3d << right_border(0), -1.0, right_border(1);

    Line3d left_border_ground = hom_ground_image.transpose() * left_border_3d;
    Line3d right_border_ground = hom_ground_image.transpose() * right_border_3d;

    double left_norm = left_border_ground.head<2>().norm();
    double right_norm = right_border_ground.head<2>().norm();

    if (left_norm < 1e-9 || right_norm < 1e-9)
      throw std::runtime_error("TraversableNode::projectImageLineToGround: degenerate ground line!");

    return {left_border_ground, right_border_ground};
  }

} // namespace traversable_node
