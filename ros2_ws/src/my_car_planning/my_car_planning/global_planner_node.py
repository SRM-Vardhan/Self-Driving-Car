import heapq
import math

import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import Point, PoseStamped


class GlobalPlannerNode(Node):

    def __init__(self):
        super().__init__('global_planner')

        # ==========================================
        # SUBSCRIBERS
        # ==========================================

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.target_sub = self.create_subscription(
            Point,
            '/target_waypoint',
            self.target_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # ==========================================
        # PUBLISHER
        # ==========================================

        self.path_pub = self.create_publisher(
            Path,
            '/global_path',
            10
        )

        # ==========================================
        # CURRENT STATE
        # ==========================================

        self.current_map = None
        self.current_pose = None
        self.target_waypoint = None

        # Occupancy threshold
        self.occupied_threshold = 50

        self.get_logger().info(
            'A* Global Planner started.'
        )

    # ==========================================
    # MAP CALLBACK
    # ==========================================

    def map_callback(self, msg):

        self.current_map = msg

        if (
            self.current_pose is not None
            and self.target_waypoint is not None
        ):
            self.plan_path()

    # ==========================================
    # ODOM CALLBACK
    # ==========================================

    def odom_callback(self, msg):

        self.current_pose = msg.pose.pose

        if (
            self.current_map is not None
            and self.target_waypoint is not None
        ):
            self.plan_path()

    # ==========================================
    # TARGET CALLBACK
    # ==========================================

    def target_callback(self, msg):

        self.target_waypoint = msg

        if (
            self.current_map is not None
            and self.current_pose is not None
        ):
            self.plan_path()

    # ==========================================
    # WORLD → MAP CELL
    # ==========================================

    def world_to_map(self, x, y):

        resolution = self.current_map.info.resolution
        origin_x = self.current_map.info.origin.position.x
        origin_y = self.current_map.info.origin.position.y

        map_x = int(
            math.floor((x - origin_x) / resolution)
        )

        map_y = int(
            math.floor((y - origin_y) / resolution)
        )

        return map_x, map_y

    # ==========================================
    # MAP CELL → WORLD
    # ==========================================

    def map_to_world(self, cell_x, cell_y):

        resolution = self.current_map.info.resolution
        origin_x = self.current_map.info.origin.position.x
        origin_y = self.current_map.info.origin.position.y

        world_x = (
            origin_x
            + (cell_x + 0.5) * resolution
        )

        world_y = (
            origin_y
            + (cell_y + 0.5) * resolution
        )

        return world_x, world_y

    # ==========================================
    # CHECK MAP BOUNDS
    # ==========================================

    def is_inside_map(self, x, y):

        width = self.current_map.info.width
        height = self.current_map.info.height

        return (
            0 <= x < width
            and
            0 <= y < height
        )

    # ==========================================
    # CHECK WHETHER CELL IS FREE
    # ==========================================

    def is_free(self, x, y):

        if not self.is_inside_map(x, y):
            return False

        width = self.current_map.info.width

        index = y * width + x

        value = self.current_map.data[index]

        # Unknown cells are not traversed
        if value < 0:
            return False

        # Occupied cells are not traversed
        if value >= self.occupied_threshold:
            return False

        return True

    # ==========================================
    # FIND NEAREST FREE CELL
    # ==========================================

    def find_nearest_free_cell(
        self,
        cell,
        search_radius=10
    ):

        start_x, start_y = cell

        if self.is_free(start_x, start_y):
            return cell

        best_cell = None
        best_distance = float('inf')

        for radius in range(
            1,
            search_radius + 1
        ):

            for dx in range(
                -radius,
                radius + 1
            ):

                for dy in range(
                    -radius,
                    radius + 1
                ):

                    x = start_x + dx
                    y = start_y + dy

                    if not self.is_free(x, y):
                        continue

                    distance = (
                        dx * dx
                        +
                        dy * dy
                    )

                    if distance < best_distance:

                        best_distance = distance
                        best_cell = (x, y)

            if best_cell is not None:
                return best_cell

        return None

    # ==========================================
    # A* HEURISTIC
    # ==========================================

    def heuristic(self, a, b):

        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])

        return math.sqrt(
            dx * dx + dy * dy
        )

    # ==========================================
    # A* PATH SEARCH
    # ==========================================

    def a_star(self, start, goal):

        open_set = []

        heapq.heappush(
            open_set,
            (
                0.0,
                start
            )
        )

        came_from = {}

        g_score = {
            start: 0.0
        }

        # 8-connected grid
        neighbors = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),

            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1)
        ]

        while open_set:

            _, current = heapq.heappop(
                open_set
            )

            if current == goal:

                return self.reconstruct_path(
                    came_from,
                    current
                )

            current_g = g_score[current]

            for dx, dy in neighbors:

                neighbor = (
                    current[0] + dx,
                    current[1] + dy
                )

                if not self.is_free(
                    neighbor[0],
                    neighbor[1]
                ):
                    continue

                # Prevent diagonal movement
                # through obstacle corners.
                if dx != 0 and dy != 0:

                    if not self.is_free(
                        current[0] + dx,
                        current[1]
                    ):
                        continue

                    if not self.is_free(
                        current[0],
                        current[1] + dy
                    ):
                        continue

                if dx != 0 and dy != 0:
                    movement_cost = math.sqrt(2.0)
                else:
                    movement_cost = 1.0

                tentative_g = (
                    current_g
                    + movement_cost
                )

                if (
                    neighbor not in g_score
                    or
                    tentative_g < g_score[neighbor]
                ):

                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g

                    f_score = (
                        tentative_g
                        +
                        self.heuristic(
                            neighbor,
                            goal
                        )
                    )

                    heapq.heappush(
                        open_set,
                        (
                            f_score,
                            neighbor
                        )
                    )

        return None

    # ==========================================
    # RECONSTRUCT A* PATH
    # ==========================================

    def reconstruct_path(
        self,
        came_from,
        current
    ):

        path = [current]

        while current in came_from:

            current = came_from[current]
            path.append(current)

        path.reverse()

        return path

    # ==========================================
    # CALCULATE YAW
    # ==========================================

    def calculate_yaw(
        self,
        current,
        next_point
    ):

        dx = (
            next_point[0]
            -
            current[0]
        )

        dy = (
            next_point[1]
            -
            current[1]
        )

        return math.atan2(
            dy,
            dx
        )

    # ==========================================
    # YAW → QUATERNION
    # ==========================================

    def yaw_to_quaternion(self, yaw):

        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)

        return qz, qw

    # ==========================================
    # CONVERT CELL PATH TO ROS PATH
    # ==========================================

    def create_ros_path(
        self,
        cell_path
    ):

        path_msg = Path()

        path_msg.header.stamp = (
            self.get_clock().now().to_msg()
        )

        path_msg.header.frame_id = (
            self.current_map.header.frame_id
            if self.current_map.header.frame_id
            else 'map'
        )

        for i, cell in enumerate(cell_path):

            x, y = self.map_to_world(
                cell[0],
                cell[1]
            )

            pose = PoseStamped()

            pose.header = path_msg.header

            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0

            if i < len(cell_path) - 1:

                next_x, next_y = (
                    self.map_to_world(
                        cell_path[i + 1][0],
                        cell_path[i + 1][1]
                    )
                )

                yaw = self.calculate_yaw(
                    (x, y),
                    (next_x, next_y)
                )

            elif i > 0:

                previous_x, previous_y = (
                    self.map_to_world(
                        cell_path[i - 1][0],
                        cell_path[i - 1][1]
                    )
                )

                yaw = self.calculate_yaw(
                    (previous_x, previous_y),
                    (x, y)
                )

            else:

                yaw = 0.0

            qz, qw = self.yaw_to_quaternion(
                yaw
            )

            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw

            path_msg.poses.append(
                pose
            )

        return path_msg

    # ==========================================
    # MAIN PLANNING FUNCTION
    # ==========================================

    def plan_path(self):

        if self.current_map is None:
            return

        if self.current_pose is None:
            return

        if self.target_waypoint is None:
            return

        # Current vehicle position
        start_world_x = (
            self.current_pose.position.x
        )

        start_world_y = (
            self.current_pose.position.y
        )

        # Target waypoint
        goal_world_x = (
            self.target_waypoint.x
        )

        goal_world_y = (
            self.target_waypoint.y
        )

        # World → grid
        start = self.world_to_map(
            start_world_x,
            start_world_y
        )

        goal = self.world_to_map(
            goal_world_x,
            goal_world_y
        )

        # Check bounds
        if not self.is_inside_map(
            start[0],
            start[1]
        ):

            self.get_logger().warn(
                'Current vehicle position is '
                'outside the map.'
            )

            return

        if not self.is_inside_map(
            goal[0],
            goal[1]
        ):

            self.get_logger().warn(
                'Target waypoint is outside '
                'the map.'
            )

            return

        # Find nearby free cells if necessary
        free_start = (
            self.find_nearest_free_cell(
                start
            )
        )

        free_goal = (
            self.find_nearest_free_cell(
                goal
            )
        )

        if free_start is None:

            self.get_logger().warn(
                'Could not find a free cell '
                'near the vehicle.'
            )

            return

        if free_goal is None:

            self.get_logger().warn(
                'Could not find a free cell '
                'near the target.'
            )

            return

        # Run A*
        cell_path = self.a_star(
            free_start,
            free_goal
        )

        if cell_path is None:

            self.get_logger().warn(
                'A* could not find a path '
                'to the target.'
            )

            return

        # Convert to ROS Path
        ros_path = self.create_ros_path(
            cell_path
        )

        # Publish
        self.path_pub.publish(
            ros_path
        )

        self.get_logger().info(
            f'A* path published: '
            f'{len(cell_path)} points'
        )


def main(args=None):

    rclpy.init(args=args)

    node = GlobalPlannerNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()