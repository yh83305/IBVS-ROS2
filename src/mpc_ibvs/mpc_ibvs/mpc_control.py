#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Twist
from scipy.optimize import minimize
from custom_msgs.msg import DetectionResult

class MPCController(Node):
    def __init__(self):
        super().__init__('mpc_controller')
        self.detection_sub = self.create_subscription(DetectionResult, '/detection_result', self.detection_callback, 10)
        self.twist_pub = self.create_publisher(Twist, '/yocs_cmd_vel_mux/move_base/cmd_vel', 10)

        self.fx = 422.11907958984375
        self.fy = 422.11907958984375
        self.cx = 424.567138671875
        self.cy = 233.30215454101562
        self.image_width = 848
        self.image_height = 640
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        self.F = np.zeros((8, 8))
        for i in range(0, 8, 2): 
            self.F[i, i] = self.fx
        for i in range(1, 8, 2):
            self.F[i, i] = self.fy

        self.F_inv = np.linalg.inv(self.F)

        self.Q = np.eye(8)  # 加权矩阵 (8x8单位矩阵)
        self.R = 100000 * np.eye(2)  # 正则化权重
        self.Np = 4
        self.Ts = 0.1
        self.tau_last = np.full((2, 1), 0.1)

        self.desired_uv = np.array([
                                    [139],
                                    [244],

                                    [453],
                                    [246],

                                    [138],
                                    [401],

                                    [453],
                                    [411]
                                    ])
        
        self.s = None
        self.Z = None

        self.timer = self.create_timer(0.1, self.mpc_controller)  # 10Hz
        self.get_logger().info("MPC controller Init")

    def detection_callback(self, msg):
        self.s = np.array(msg.s).reshape(-1, 1)
        self.Z = msg.z

    def interaction_matrix(self, s, Z, K):
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        u = s[::2]
        v = s[1::2]
        x = (u - cx) / fx
        y = (v - cy) / fy
        Ls = []
        for i in range(len(x)):
            Lsi = np.array([[x[i,0] / Z, -(1 + x[i,0] ** 2)],
                            [y[i,0] / Z, -x[i,0] * y[i,0]]])
            Ls.append(Lsi)
        L = np.vstack(Ls)
        return L

    def mpc_controller(self):
        if self.s is None or self.Z is None:
            return
        
        # 定义视野约束
        def fov_constraints(tau_seq, s, Z):
            tau_seq = np.asarray(tau_seq)
            tau_seq = tau_seq.flatten()
            c = []
            s_pred = s
            for j in range(self.Np):
                Ls = self.interaction_matrix(s_pred, Z, self.K)
                tau = tau_seq[2*j:2*(j+1)].reshape(-1, 1)
                s_pred = s_pred + self.Ts * (self.F @ Ls @ tau)
                u = s_pred[::2]
                v = s_pred[1::2]
                c.extend([u + 0,
                        -u + (self.image_width - 10),
                        v + 0,
                        -v + (self.image_height - 10)])
            c = np.concatenate(c)
            return c.flatten()

        # 定义成本函数
        def cost_function(tau_seq, s, Z):
            tau_seq = np.asarray(tau_seq)
            tau_seq = tau_seq.flatten()

            J1 = 0
            J2 = 0
            s_pred = s
            for i in range(self.Np):
                L = self.interaction_matrix(s_pred, Z, self.K)
                tau = tau_seq[2*i:2*(i+1)].reshape(-1, 1)
                s_pred = s_pred + self.Ts * (self.F @ L @ tau)
                e = s_pred - self.desired_uv.reshape(-1, 1)
                J1 += e.T @ self.Q @ e
                J2 += tau.T @ self.R @ tau
            
            J = J1 + J2
            return J

        # 初始猜测 (6*Np 维向量)
        tau_seq0 = np.tile(self.tau_last, self.Np)

        constraints = [
            {'type': 'ineq', 'fun': fov_constraints, 'args': (self.s, self.Z)},
            {'type': 'ineq', 'fun': lambda tau_seq: 0.2 - tau_seq[0::6]},  # vz <= 0.5
            {'type': 'ineq', 'fun': lambda tau_seq: 0.2 + tau_seq[0::6]},  # vz >= -0.5
            {'type': 'ineq', 'fun': lambda tau_seq: 0.1 - tau_seq[1::6]},  # wy <= 0.3
            {'type': 'ineq', 'fun': lambda tau_seq: 0.1 + tau_seq[1::6]},  # wy >= -0.3
        ]

        res = minimize(cost_function, tau_seq0, args=(self.s, self.Z), method='SLSQP', constraints=constraints, options={'maxiter': 500, 'ftol': 1e-6})

        tau = res.x[:2]
        self.tau_last = tau
        self.publish_twist(tau)
        self.get_logger().info("twist_wv: linear_x = %f, angular_z = %f" % (tau[0], tau[1]))

    def publish_twist(self, twist_car):
        twist_msg = Twist()
        twist_msg.linear.x = np.clip(twist_car[0], -0.2, 0.2)   # v_x
        twist_msg.angular.z = np.clip(-twist_car[1], -0.1, 0.1)  # w_z
        self.twist_pub.publish(twist_msg)

def main(args=None):
    rclpy.init(args=args)
    mpc_controller = MPCController()
    rclpy.spin(mpc_controller)
    mpc_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()