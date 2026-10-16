#!/usr/bin/python3
import math

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from ament_index_python.packages import get_package_share_path


class MP4Publisher(Node):
    def __init__(self):
        super().__init__("mp4_publisher")

        self.is_loop = self.declare_parameter('make_loop', False).value
        video_path = self.declare_parameter('video_path', 'vineyard/video1.mp4').value
        image_topic_name = self.declare_parameter('image_topic_name', '/fake_camera/image_raw').value

        share_path = get_package_share_path('traversable_corridor')
        video_path = share_path / 'video' / video_path
        
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Can't open video, video path: {video_path}")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0

        self.bridge = CvBridge()
        self.pub = self.create_publisher(RosImage, image_topic_name, 1)
        self.done = False
        self.timer = self.create_timer(1.0 / fps, self.publish_frame)

    def publish_frame(self):
        # Не пропускаем начало видео, пока inference не подключился.
        if self.pub.get_subscription_count() == 0:
            return

        ok, frame = self.cap.read()
        if not ok and self.is_loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            
        if not ok:
            self.get_logger().error("Can't read begin of video!")
            self.timer.cancel()
            self.done = True
            return

        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        self.pub.publish(msg)


def main():
  rclpy.init()
  node = None
  try:
      node = MP4Publisher()
      while rclpy.ok() and not node.done:
          rclpy.spin_once(node)
  except KeyboardInterrupt:
      pass
  finally:
      if node is not None:
          node.cap.release()
          node.destroy_node()
      if rclpy.ok():
          rclpy.shutdown()


if __name__ == "__main__":
  main()
