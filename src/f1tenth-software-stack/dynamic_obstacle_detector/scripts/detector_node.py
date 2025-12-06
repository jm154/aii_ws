#!/usr/bin/env python3

# Import necessary libraries
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

import tf2_ros
import tf_transformations # Uses the 'tf-transformations' pip package
import numpy as np
from sklearn.cluster import DBSCAN # Uses the 'scikit-learn' pip package

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# Attempt to import range_libc (Must be installed/built in the ROS 2 workspace)
try:
    import range_libc
except ImportError:
    # Log error and potentially raise an exception or handle gracefully
    # For now, just print a warning. The node will likely fail later.
    print(
        "\n"
        "********************************************************************************\n"
        "Failed to import 'range_libc'. \n"
        "Ensure 'range_libc' (often provided by 'f1tenth_gym_ros' or similar) \n"
        "is correctly built and sourced in your ROS 2 workspace (e.g., /sim_ws).\n"
        "Run 'colcon build' and 'source install/setup.bash'.\n"
        "********************************************************************************\n"
    )
    # Depending on requirements, you might want to exit or raise here
    # raise ImportError("range_libc not found, cannot proceed.")

# Helper class to track detected objects (simplified version without Kalman Filter)
class TrackedObject:
    """Represents a tracked dynamic object with simple velocity estimation."""
    def __init__(self, cluster_id, centroid, clock):
        self.id = cluster_id
        self.centroid = np.array(centroid) # Store as numpy array [x, y]
        self.velocity = np.array([0.0, 0.0]) # Estimated velocity [vx, vy]
        self.last_update_time = clock.now() # ROS 2 Time object

    def update(self, new_centroid, now):
        """Updates the object's state with a new centroid measurement."""
        new_centroid = np.array(new_centroid)
        dt_duration = now - self.last_update_time
        dt = dt_duration.nanoseconds / 1e9 # Convert duration to seconds

        # Avoid division by zero or too frequent updates
        if dt > 0.01: # Update at max 100 Hz
            # Simple velocity estimation (difference / time)
            self.velocity = (new_centroid - self.centroid) / dt
            self.centroid = new_centroid
            self.last_update_time = now
            # In a real system, a Kalman Filter would predict and update state here

# Main ROS 2 Node class
class DynamicObjectDetector(Node):
    """
    Detects dynamic objects by comparing LiDAR scans to a static map.
    Uses range_libc for fast ray-casting and DBSCAN for clustering.
    """
    def __init__(self):
        super().__init__('dynamic_detector_node') # Node name
        self.get_logger().info("Initializing Dynamic Object Detector Node...")

        # Initialize range_libc object (will be set in map_callback)
        self.range_method = None
        self.map_info = None # To store map metadata

        # TF2 listener setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Declare and get parameters
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('laser_frame', 'laser') # Default, remap in launch file
        self.declare_parameter('dynamic_threshold', 0.3) # [m] Diff threshold
        self.declare_parameter('cluster_eps', 0.4) # [m] DBSCAN neighborhood radius
        self.declare_parameter('cluster_min_samples', 3) # Min points for DBSCAN cluster

        self.map_frame = self.get_parameter('map_frame').value
        # Note: laser_frame parameter isn't directly used here,
        # as we get the frame_id from the LaserScan message header.
        # It's useful for configuration clarity or potential future use.
        self.DYNAMIC_THRESHOLD = self.get_parameter('dynamic_threshold').value
        self.CLUSTER_EPS = self.get_parameter('cluster_eps').value
        self.CLUSTER_MIN_SAMPLES = self.get_parameter('cluster_min_samples').value

        # Object tracking state
        self.tracked_objects = {} # Dictionary: {id: TrackedObject instance}
        self.next_object_id = 0

        # Publisher for visualization markers (clusters and velocity arrows)
        self.marker_pub = self.create_publisher(MarkerArray, '/detected_objects', 10) # Remap in launch file if needed

        # Subscribers
        # Map subscription: Use TRANSIENT_LOCAL durability to get the last published map
        qos_map = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',             # Standard map topic
            self.map_callback,
            qos_profile=qos_map # Ensure we get the map even if we start after map_server
        )

        # Scan subscription: Use BEST_EFFORT for potentially lossy sensor data
        qos_scan = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',            # Remap in launch file to specific vehicle scan
            self.scan_callback,
            qos_scan
        )

        self.get_logger().info("Dynamic Object Detector Node initialized. Waiting for map...")

    def map_callback(self, msg: OccupancyGrid):
        """Callback for receiving the static map. Initializes range_libc."""
        # Only initialize once
        if self.range_method is not None:
            return

        self.map_info = msg.info
        self.get_logger().info(
            f"Received map: {self.map_info.width}x{self.map_info.height} @ {self.map_info.resolution:.3f} m/cell"
        )
        try:
            # Initialize range_libc's ray marching GPU accelerated method
            # This requires the OccupancyGrid message and map resolution
            self.range_method = range_libc.PyRayMarching(msg, self.map_info.resolution)
            self.get_logger().info("range_libc initialized successfully.")
        except Exception as e:
            self.get_logger().error(f"Error initializing range_libc: {e}")
            # Consider shutting down or entering a safe state if range_libc is critical
            # For now, we'll just log the error and the scan_callback will wait.

    def scan_callback(self, scan_msg: LaserScan):
        """Main callback processing incoming laser scans."""
        # Wait until map and range_libc are ready
        if self.range_method is None:
            self.get_logger().warn_throttle(
                self.get_clock(), 5000, # Log every 5 seconds
                "Waiting for map to initialize range_libc..."
            )
            return

        # 1. Get the sensor pose in the map frame at the time of the scan
        try:
            # Use the timestamp from the scan message header for accurate transform
            transform_stamped = self.tf_buffer.lookup_transform(
                self.map_frame,             # Target frame
                scan_msg.header.frame_id,   # Source frame (from scan message)
                scan_msg.header.stamp,      # Time point of the scan
                timeout=Duration(seconds=0.1) # Wait up to 0.1s for transform
            )
            transform = transform_stamped.transform
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn_throttle(
                self.get_clock(), 2000, # Log every 2 seconds
                f"TF lookup failed from '{scan_msg.header.frame_id}' to '{self.map_frame}': {e}"
            )
            return

        # Extract laser pose (x, y, yaw) from the transform
        laser_x = transform.translation.x
        laser_y = transform.translation.y
        quat = transform.rotation
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        _, _, yaw = tf_transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])

        # 2. Calculate Expected Ranges using range_libc
        # Get all angles relative to the laser frame
        num_ranges = len(scan_msg.ranges)
        angles_laser_frame = scan_msg.angle_min + np.arange(num_ranges) * scan_msg.angle_increment
        # Convert angles to the map frame by adding the laser's yaw
        angles_map_frame = angles_laser_frame + yaw

        # Call range_libc to get expected distances for all angles at once
        # Input: laser's (x, y) in map frame, array of angles in map frame
        # Output: array of expected ranges
        try:
            expected_ranges = self.range_method.calc_range_many(laser_x, laser_y, angles_map_frame)
        except Exception as e:
             self.get_logger().error_throttle(
                 self.get_clock(), 5000,
                 f"range_libc calc_range_many failed: {e}"
             )
             return

        # 3. Compare Actual vs. Expected Ranges and Identify Dynamic Points
        dynamic_points_world = [] # List to store coordinates of dynamic points

        actual_ranges = np.array(scan_msg.ranges)

        for i in range(num_ranges):
            actual_range = actual_ranges[i]

            # Filter out invalid range readings (inf, nan, too short, too long)
            if not np.isfinite(actual_range) or \
               actual_range < scan_msg.range_min or \
               actual_range > scan_msg.range_max:
                continue

            expected_range = expected_ranges[i]

            # Core Logic: If actual range is significantly shorter than expected,
            # it's likely hitting a dynamic object not present in the static map.
            if actual_range < expected_range - self.DYNAMIC_THRESHOLD:
                # Calculate the world coordinates (x, y) of this dynamic point
                point_angle_laser = angles_laser_frame[i] # Angle relative to laser

                # Point coordinates in the laser frame
                local_x = actual_range * np.cos(point_angle_laser)
                local_y = actual_range * np.sin(point_angle_laser)

                # Transform point from laser frame to map frame using current laser pose
                # (Standard 2D rotation + translation)
                world_x = laser_x + (local_x * np.cos(yaw) - local_y * np.sin(yaw))
                world_y = laser_y + (local_x * np.sin(yaw) + local_y * np.cos(yaw))

                dynamic_points_world.append([world_x, world_y])

        # If no dynamic points detected, clear markers and return
        if not dynamic_points_world:
            self.publish_markers([]) # Publish empty list to clear previous markers
            return

        # 4. Cluster Dynamic Points using DBSCAN
        X = np.array(dynamic_points_world) # Convert list of points to NumPy array

        # Apply DBSCAN: eps=max distance between points for one to be considered as in the neighborhood of the other.
        # min_samples=number of samples in a neighborhood for a point to be considered as a core point.
        db = DBSCAN(eps=self.CLUSTER_EPS, min_samples=self.CLUSTER_MIN_SAMPLES).fit(X)

        # Extract cluster labels (-1 means noise)
        labels = db.labels_
        unique_labels = set(labels)

        # Calculate the centroid (average point) for each valid cluster
        current_clusters = {} # Dictionary: {cluster_label: centroid_np_array}
        if len(unique_labels) > 0 and not (len(unique_labels) == 1 and -1 in unique_labels) :
             self.get_logger().debug(f"DBSCAN found {len(unique_labels)- (1 if -1 in unique_labels else 0)} clusters from {len(X)} points.")
        for k in unique_labels:
            if k == -1:
                # Skip noise points
                continue
            # Get points belonging to this cluster
            cluster_mask = (labels == k)
            cluster_points = X[cluster_mask]
            # Calculate the centroid (mean x, mean y)
            centroid = np.mean(cluster_points, axis=0)
            current_clusters[k] = centroid # Store centroid

        # 5. Update Object Tracking (Data Association and State Update)
        self.update_tracking(current_clusters)

        # 6. Publish Visualization Markers
        self.publish_markers(self.tracked_objects.values())

    def update_tracking(self, current_clusters):
        """Associates current clusters with previously tracked objects and updates state."""
        # Simple association: Match current clusters to nearest tracked object centroid
        # More robust methods: Hungarian algorithm, JPDAF

        matched_existing_ids = set() # Keep track of which tracked objects got matched
        new_tracked_objects = {} # Build the next state of tracked objects

        now = self.get_clock().now() # Current time for velocity calculation

        for cluster_label, current_centroid in current_clusters.items():
            best_match_id = None
            # Set a maximum distance for matching to avoid incorrect associations
            min_dist_threshold = 0.8 # [m] Max distance to associate

            # Find the closest *unmatched* tracked object from the previous step
            for obj_id, tracked_obj in self.tracked_objects.items():
                # Skip if this tracked object has already been matched
                if obj_id in matched_existing_ids:
                    continue

                dist = np.linalg.norm(tracked_obj.centroid - current_centroid)

                if dist < min_dist_threshold:
                    min_dist_threshold = dist # Update threshold to find the absolute closest
                    best_match_id = obj_id

            if best_match_id is not None:
                # Found a match: Update the existing tracked object
                self.tracked_objects[best_match_id].update(current_centroid, now)
                # Add the updated object to the next state dictionary
                new_tracked_objects[best_match_id] = self.tracked_objects[best_match_id]
                # Mark this ID as matched
                matched_existing_ids.add(best_match_id)
                self.get_logger().debug(f"Matched cluster {cluster_label} to object {best_match_id}")
            else:
                # No match found: Create a new tracked object
                new_id = self.next_object_id
                new_object = TrackedObject(new_id, current_centroid, self.get_clock())
                new_tracked_objects[new_id] = new_object
                self.next_object_id += 1 # Increment ID for the next new object
                self.get_logger().debug(f"Created new object {new_id} for cluster {cluster_label}")

        # Update the node's tracked object list. Objects that weren't matched disappear.
        self.tracked_objects = new_tracked_objects

        # Optional: Implement logic to remove objects that haven't been seen for a while

    def publish_markers(self, objects):
        """Publishes markers for RViz visualization (centroids and velocity arrows)."""
        marker_array = MarkerArray()
        now_stamp = self.get_clock().now().to_msg() # Timestamp for all markers

        # Action DELETEALL: Clear all markers from previous publications in this namespace
        # This is simpler than tracking individual marker lifetimes
        delete_marker = Marker()
        delete_marker.header.frame_id = self.map_frame
        delete_marker.header.stamp = now_stamp
        delete_marker.ns = "dynamic_objects" # Namespace for centroids
        delete_marker.id = 0 # ID doesn't matter much for DELETEALL
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        delete_marker_vel = Marker()
        delete_marker_vel.header.frame_id = self.map_frame
        delete_marker_vel.header.stamp = now_stamp
        delete_marker_vel.ns = "dynamic_velocities" # Namespace for velocities
        delete_marker_vel.id = 1
        delete_marker_vel.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker_vel)

        # Create markers for each currently tracked object
        for obj in objects:
            # 1. Centroid Marker (Cylinder)
            marker = Marker()
            marker.header.frame_id = self.map_frame
            marker.header.stamp = now_stamp
            marker.ns = "dynamic_objects" # Namespace
            marker.id = obj.id # Unique ID for this object's marker
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            # Pose: Position object centroid, default orientation (Quaternion w=1)
            marker.pose.position.x = float(obj.centroid[0])
            marker.pose.position.y = float(obj.centroid[1])
            marker.pose.position.z = 0.15 # Slightly above ground
            marker.pose.orientation.w = 1.0
            # Scale: Diameter and height of the cylinder
            marker.scale.x = 0.5 # Diameter
            marker.scale.y = 0.5 # Diameter
            marker.scale.z = 0.3 # Height
            # Color: Orange, mostly opaque
            marker.color.a = 0.8 # Alpha (transparency)
            marker.color.r = 1.0 # Red
            marker.color.g = 0.5 # Green
            marker.color.b = 0.0 # Blue
            # marker.lifetime = Duration(seconds=0.5).to_msg() # Auto-delete after 0.5s if not updated
            marker_array.markers.append(marker)

            # 2. Velocity Marker (Arrow) - Only if velocity is significant
            velocity_magnitude = np.linalg.norm(obj.velocity)
            if velocity_magnitude > 0.1: # Threshold to avoid tiny arrows for noise
                vel_marker = Marker()
                vel_marker.header.frame_id = self.map_frame
                vel_marker.header.stamp = now_stamp
                vel_marker.ns = "dynamic_velocities" # Separate namespace
                vel_marker.id = obj.id # Match object ID
                vel_marker.type = Marker.ARROW
                vel_marker.action = Marker.ADD

                # Points: Defines the start and end of the arrow
                start_point = Point()
                start_point.x = float(obj.centroid[0])
                start_point.y = float(obj.centroid[1])
                start_point.z = 0.3 # Arrow starts slightly higher

                end_point = Point()
                # End point represents position after 0.5 seconds at current velocity
                arrow_scale_factor = 0.5
                end_point.x = float(obj.centroid[0] + obj.velocity[0] * arrow_scale_factor) # Scale arrow length
                end_point.y = float(obj.centroid[1] + obj.velocity[1] * arrow_scale_factor)
                end_point.z = 0.3 # Keep arrow flat

                vel_marker.points = [start_point, end_point]

                # Scale: Arrow dimensions (shaft diameter, head diameter, head length)
                vel_marker.scale.x = 0.05 # Shaft diameter
                vel_marker.scale.y = 0.1  # Head diameter
                vel_marker.scale.z = 0.1  # Head length (usually ignored for ARROW type, but set anyway)

                # Color: Green, fully opaque
                vel_marker.color.a = 1.0
                vel_marker.color.r = 0.0
                vel_marker.color.g = 1.0
                vel_marker.color.b = 0.0
                # vel_marker.lifetime = Duration(seconds=0.5).to_msg()
                marker_array.markers.append(vel_marker)

        # Publish the complete array of markers
        self.marker_pub.publish(marker_array)

# Standard Python entry point
def main(args=None):
    rclpy.init(args=args) # Initialize ROS 2 communications
    try:
        detector_node = DynamicObjectDetector() # Create the node instance
        rclpy.spin(detector_node) # Keep the node alive and processing callbacks
    except KeyboardInterrupt:
        pass # Allow Ctrl+C to gracefully exit
    except Exception as e:
         print(f"Error in node execution: {e}")
    finally:
        # Clean up resources even if errors occur
        if 'detector_node' in locals() and rclpy.ok():
            detector_node.destroy_node() # Properly shut down the node
        if rclpy.ok():
             rclpy.shutdown() # Shut down ROS 2 communications

# This ensures main() is called when the script is executed directly
if __name__ == '__main__':
    main()
