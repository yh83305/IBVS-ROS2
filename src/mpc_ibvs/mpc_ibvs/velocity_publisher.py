import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class VelocityPublisher(Node):

    def __init__(self):
        super().__init__('velocity_publisher')
        self.publisher_ = self.create_publisher(Twist, '/yocs_cmd_vel_mux/move_base/cmd_vel', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info('Velocity Publisher Node has been started.')

    def timer_callback(self):
        msg = Twist()
        msg.linear.x = 0.01  # Set linear velocity in x direction
        msg.angular.z = 0.01  # Set angular velocity around z axis
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: Linear Velocity: %f, Angular Velocity: %f' % (msg.linear.x, msg.angular.z))

def main(args=None):
    rclpy.init(args=args)
    velocity_publisher = VelocityPublisher()
    rclpy.spin(velocity_publisher)
    velocity_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()