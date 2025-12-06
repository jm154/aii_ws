#!/usr/bin/env python3
import rclpy, math
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import tf2_ros
from functools import partial

#!/usr/bin/env python3
import rclpy, math
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import tf2_ros
from functools import partial

class MultiOdomToTF(Node):
    def __init__(self, name='multi_odom_to_tf'):
        super().__init__(name)
        
        # --- Params ---
        self.declare_parameter('world_frame', 'odom')
        self.declare_parameter('vehicle_names', ['ego_racecar', 'opp_racecar'])
        
        use_sim_time_default = True
        try:
            if not self.has_parameter('use_sim_time'):
                self.declare_parameter('use_sim_time', use_sim_time_default)
        except Exception:
            pass
        
        use_sim_time_param = self.get_parameter('use_sim_time')
        use_sim_time = (use_sim_time_param.type_ != Parameter.Type.NOT_SET
                        and use_sim_time_param.get_parameter_value().bool_value) or use_sim_time_default
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, use_sim_time)])

        # --- Values ---
        self.world_frame = self.get_parameter('world_frame').get_parameter_value().string_value
        self.vehicle_names = self.get_parameter('vehicle_names').get_parameter_value().string_array_value

        self.br = tf2_ros.TransformBroadcaster(self)
        self.subs = []

        for vehicle_name in self.vehicle_names:
            odom_topic = f'/{vehicle_name}/odom'
            base_frame = f'{vehicle_name}/base_link'
            
            # partial을 사용하여 각 콜백이 자신의 base_frame을 기억하도록 함
            callback = partial(self.odom_callback, base_frame=base_frame)
            
            sub = self.create_subscription(Odometry, odom_topic, callback, 10)
            self.subs.append(sub)
            self.get_logger().info(f"Bridging {odom_topic} -> TF '{self.world_frame}' to '{base_frame}'")

        self.get_logger().info(f"Node '{name}' is ready for {len(self.vehicle_names)} vehicles.")
        self.get_logger().info(f"use_sim_time = {use_sim_time}")

    def odom_callback(self, msg: Odometry, base_frame: str):
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = self.world_frame
        t.child_frame_id = base_frame
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation
        self.br.sendTransform(t)
        self.get_logger().debug(f"Published transform: {t.header.stamp.sec}.{t.header.stamp.nanosec} {t.header.frame_id} -> {t.child_frame_id}")

def main():
    rclpy.init()
    node = MultiOdomToTF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()
