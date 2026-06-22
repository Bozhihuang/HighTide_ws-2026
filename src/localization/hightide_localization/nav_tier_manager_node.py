#!/usr/bin/env python3
"""
Navigation Tier Manager — Monitors ZED tracking quality and selects nav tier.

Tier 1 (VSLAM):  ZED confidence > 80%, spatial memory active
Tier 2 (VIO):    ZED confidence 30-80% or no spatial memory
Tier 3 (Dead Reckoning): ZED confidence < 30% or data stale
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from hightide_interfaces.msg import NavigationTier
from hightide_interfaces.srv import SetNavigationTier
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy



class NavTierManagerNode(Node):
    """Manages which navigation tier is active based on sensor quality."""

    def __init__(self):
        super().__init__('nav_tier_manager_node')

        self.declare_parameter('vslam_confidence_threshold', 0.80)
        self.declare_parameter('vio_confidence_threshold', 0.30)
        self.declare_parameter('stale_timeout_sec', 1.0)
        self.declare_parameter('publish_rate', 5.0)

        self.vslam_thresh = self.get_parameter('vslam_confidence_threshold').value
        self.vio_thresh = self.get_parameter('vio_confidence_threshold').value
        self.stale_timeout = self.get_parameter('stale_timeout_sec').value
        qos_profile = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        depth=10
        )

        # State
        self.current_tier = NavigationTier.VIO  # Default
        self.tracking_confidence = 0.0
        self.forced_tier = None
        self.last_odom_time = None
        self.last_imu_time = None
        self.fog_available = False
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/mavros/zed/odom',
            self._odom_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/mavros/imu/data', self._imu_callback, qos_profile)

        # Publisher
        self.tier_pub = self.create_publisher(
            NavigationTier, '/hightide/navigation_tier', 10)

        # Service
        self.set_tier_srv = self.create_service(
            SetNavigationTier, '/hightide/set_navigation_tier',
            self._set_tier_service)

        # Timer
        period = 1.0 / self.get_parameter('publish_rate').value
        self.timer = self.create_timer(period, self._evaluate_tier)

        self.get_logger().info('Nav Tier Manager started')

    def _odom_callback(self, msg: Odometry):
        """Track ZED odometry freshness and covariance."""
        self.last_odom_time = self.get_clock().now()

        # Use position covariance as a proxy for tracking confidence
        # Lower covariance = better tracking
        cov = msg.pose.covariance
        avg_pos_cov = (abs(cov[0]) + abs(cov[7]) + abs(cov[14])) / 3.0

        # Map covariance to confidence (heuristic):
        # Low cov (< 0.01) → high confidence (~1.0)
        # High cov (> 1.0) → low confidence (~0.0)
        if avg_pos_cov < 0.001:
            self.tracking_confidence = 1.0
        elif avg_pos_cov > 1.0:
            self.tracking_confidence = 0.0
        else:
            self.tracking_confidence = max(0.0, 1.0 - avg_pos_cov)

    def _imu_callback(self, msg: Imu):
        """Track FOG/IMU data availability."""
        self.last_imu_time = self.get_clock().now()
        self.fog_available = True

    def _set_tier_service(self, request, response):
        """Force a specific navigation tier (0 = auto)."""
        if request.tier == 0:
            self.forced_tier = None
            response.message = 'Navigation tier set to AUTO'
        elif request.tier in (1, 2, 3):
            self.forced_tier = request.tier
            response.message = f'Navigation tier forced to {request.tier}'
        else:
            response.success = False
            response.message = f'Invalid tier: {request.tier}'
            response.current_tier = self.current_tier
            return response

        response.success = True
        response.current_tier = self.forced_tier or self.current_tier
        self.get_logger().info(response.message)
        return response

    def _evaluate_tier(self):
        """Evaluate sensor health and select appropriate tier."""
        now = self.get_clock().now()
        status = ''

        # Check if forced
        if self.forced_tier is not None:
            self.current_tier = self.forced_tier
            status = f'FORCED to tier {self.forced_tier}'
        else:
            # Check ZED data staleness
            zed_stale = True
            if self.last_odom_time:
                dt = (now - self.last_odom_time).nanoseconds / 1e9
                zed_stale = dt > self.stale_timeout

            if zed_stale or self.tracking_confidence < self.vio_thresh:
                self.current_tier = NavigationTier.DEAD_RECKONING
                status = f'ZED {"stale" if zed_stale else "low confidence"}'
            elif self.tracking_confidence >= self.vslam_thresh:
                self.current_tier = NavigationTier.VSLAM
                status = 'High confidence tracking'
            else:
                self.current_tier = NavigationTier.VIO
                status = 'Moderate confidence tracking'

        # Publish
        msg = NavigationTier()
        msg.current_tier = self.current_tier
        msg.tracking_confidence = self.tracking_confidence
        msg.status_message = status
        self.tier_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = NavTierManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()
