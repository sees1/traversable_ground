#!/usr/bin/python3
"""ROS-side image transport; model execution lives in a separate interpreter."""
import socket
import time
import threading

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as RosImage

from inference_ipc import receive_packet, send_packet


class ImageSocketBridge(Node):
    def __init__(self):
        super().__init__('image_socket_bridge')
        self.sock = None
        self.thread = None
        self.condition = threading.Condition()
        self.stopping = False
        self.pending = None
        self.received = 0
        self.sent = 0
        self.published = 0
        try:
            self.start()
        except BaseException:
            self.destroy_node()
            raise

    def start(self):
        self.bridge = CvBridge()
        image_topic = self.declare_parameter('image_topic_name', '/camera/image_raw').value
        reliability = self.declare_parameter('input_reliability', 'best_effort').value
        if reliability not in ('reliable', 'best_effort'):
            raise ValueError('input_reliability must be reliable or best_effort')
        input_qos = QoSProfile(depth=1, reliability=(ReliabilityPolicy.RELIABLE
            if reliability == 'reliable' else ReliabilityPolicy.BEST_EFFORT))
        inference_topic = self.declare_parameter('inference_topic_name', '/inference').value
        host = self.declare_parameter('socket_host', '127.0.0.1').value
        port = self.declare_parameter('socket_port', 5055).value
        startup_timeout = float(self.declare_parameter('startup_timeout', 120.0).value)
        inference_timeout = float(self.declare_parameter('inference_timeout', 60.0).value)
        if startup_timeout <= 0 or inference_timeout <= 0:
            raise ValueError('Socket timeouts must be positive')
        deadline = time.monotonic() + startup_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Cannot connect to inference at {}:{}'.format(host, port))
            try:
                self.sock = socket.create_connection((host, port), timeout=min(1.0, remaining))
                break
            except OSError:
                time.sleep(min(.1, remaining))
        self.sock.settimeout(max(.001, deadline - time.monotonic()))
        status, _ = receive_packet(self.sock)
        if status.get('status') != 'ready':
            raise RuntimeError(status.get('error', 'Inference did not become ready'))
        self.sock.settimeout(inference_timeout)
        # Reliable outputs also serve best-effort subscribers; keep only the latest result.
        self.inf_pub = self.create_publisher(RosImage, inference_topic, 1)
        self.overlay_pub = self.create_publisher(RosImage, inference_topic + '/overlay', 1)
        self.thread = threading.Thread(target=self.infer, name='inference-transport', daemon=True)
        self.thread.start()
        self.img_sub = self.create_subscription(RosImage, image_topic, self.on_image, input_qos)
        self.create_timer(5.0, self.report_status)
        self.get_logger().info('Image socket bridge connected to {}:{}'.format(host, port))

    def report_status(self):
        self.get_logger().info(
            'Frames: ROS received={}, socket sent={}, masks published={}, input publishers={}'.format(
                self.received, self.sent, self.published, self.count_publishers(self.img_sub.topic_name)))

    def on_image(self, msg):
        self.received += 1
        with self.condition:
            self.pending = msg
            self.condition.notify()

    def infer(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.stopping or self.pending is not None)
                    if self.stopping:
                        return
                    msg, self.pending = self.pending, None
                try:
                    array = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                except Exception as exc:
                    self.get_logger().error('Cannot convert input image: {}'.format(exc))
                    continue
                height, width = array.shape[:2]
                overlay = self.overlay_pub.get_subscription_count() > 0
                send_packet(self.sock, {'width': width, 'height': height, 'overlay': overlay},
                            np.ascontiguousarray(array).tobytes())
                self.sent += 1
                status, payload = receive_packet(self.sock)
                if status.get('status') != 'ok':
                    raise RuntimeError(status.get('error', 'Invalid worker response'))
                pixels = width * height
                if (status.get('width'), status.get('height'), status.get('overlay')) != (width, height, overlay):
                    raise ValueError('Worker returned unexpected image metadata')
                if len(payload) != pixels * (4 if overlay else 1):
                    raise ValueError('Worker returned unexpected image size')
                if self.stopping:
                    return
                mask = np.frombuffer(payload[:pixels], dtype=np.uint8).reshape(height, width)
                result = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
                result.header = msg.header
                self.inf_pub.publish(result)
                self.published += 1
                if overlay:
                    rgb = np.frombuffer(payload[pixels:], dtype=np.uint8).reshape(height, width, 3)
                    result = self.bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
                    result.header = msg.header
                    self.overlay_pub.publish(result)
        except Exception as exc:
            if not self.stopping:
                self.get_logger().fatal('Inference worker failed: {}'.format(exc))
                # Stop the node rather than leave a live subscription producing no output.
                self.context.try_shutdown()

    def destroy_node(self):
        with self.condition:
            self.stopping = True
            self.pending = None
            self.condition.notify_all()
        if self.sock is not None:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self.thread is not None:
            self.thread.join()
        if self.sock is not None:
            self.sock.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ImageSocketBridge()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
