#include <memory>

#include "traversable_node/traversable_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<traversable_node::TraversableNode>());
  rclcpp::shutdown();
  return 0;
}
