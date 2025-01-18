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

        self.fx = 466.32194316
        self.fy = 422.11907958984375
        self.cx = 675.94747463
        self.cy = 508.50482252

        self.k1 = -0.284867
        self.k2 = 0.06076046
        self.r1 = 0.00007458
        self.r2 = 0.00112943

        self.image_width = 1280
        self.image_height = 1024
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        self.F = np.zeros((8, 8))
        for i in range(0, 8, 2): 
            self.F[i, i] = self.fx
        for i in range(1, 8, 2):
            self.F[i, i] = self.fy

        self.F_inv = np.linalg.inv(self.F)

        self.Q = 10 * np.eye(8)  # 加权矩阵 (8x8单位矩阵)
        self.R = 1 * np.eye(2)  # 正则化权重
        self.Np = 4
        self.Ts = 0.1
        self.tau_last = np.full((2, 1), 0.1)

        self.desired_uv = np.array([
                                    [579],
                                    [449],

                                    [800],
                                    [455],

                                    [578],
                                    [627],

                                    [798],
                                    [629]
                                    ])
        
        self.desired_xy = self.transform_uv_to_xy(self.desired_uv)
        self.s = None
        self.xy_vector = None
        self.Z = None

        self.timer = self.create_timer(0.1, self.mpc_controller)  # 10Hz
        self.get_logger().info("MPC controller Init")

    def detection_callback(self, msg):
        self.s = np.array(msg.s).reshape(-1, 1)
        self.xy_vector = self.transform_uv_to_xy(self.s)
        self.Z = msg.z

    def transform_uv_to_xy(self, uv_vector):

        uv_matrix = uv_vector.reshape(-1, 2)

        x_corrected, y_corrected = self.pixel_to_normalized_with_distortion(uv_matrix[:, 0], uv_matrix[:, 1])

        xy_vector = np.column_stack((x_corrected, y_corrected)).reshape(-1, 1)

        return xy_vector
    
    def transform_xy_to_uv(self, xy_vector):

        xy_matrix = xy_vector.reshape(-1, 2)

        u_vec, v_vec = self.normalized_to_pixel_with_distortion(xy_matrix[:, 0], xy_matrix[:, 1])

        return u_vec, v_vec


    def pixel_to_normalized_with_distortion(self, u, v):
        """
        将像素坐标 (u, v) 转换为归一化平面坐标 (x, y)，并校正畸变。

        参数:
            u (np.ndarray): 像素坐标 u，形状为 (N, 1)。
            v (np.ndarray): 像素坐标 v，形状为 (N, 1)。

        返回:
            tuple: 校正后的归一化平面坐标 (x_corrected, y_corrected)，均为形状为 (N, 1) 的 NumPy 数组。
        """
        # 转换为归一化平面坐标
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy

        # 计算半径平方
        r_squared = x**2 + y**2

        # 计算径向畸变
        radial_distortion = 1 + self.k1 * r_squared + self.k2 * r_squared**2

        # 计算切向畸变
        x_tangential = 2 * self.r1 * x * y + self.r2 * (r_squared + 2 * x**2)
        y_tangential = self.r1 * (r_squared + 2 * y**2) + 2 * self.r2 * x * y

        # 校正后的归一化平面坐标
        x_corrected = x * radial_distortion + x_tangential
        y_corrected = y * radial_distortion + y_tangential

        return x_corrected, y_corrected

    def normalized_to_pixel_with_distortion(self, x_corrected, y_corrected, max_iter=100, tol=1e-6):
        """
        将校正后的归一化平面坐标 (x_corrected, y_corrected) 逆向转换为像素坐标 (u, v)。

        参数:
            x_corrected (np.ndarray): 校正后的归一化平面坐标 x，形状为 (N, 1)。
            y_corrected (np.ndarray): 校正后的归一化平面坐标 y，形状为 (N, 1)。
            max_iter (int): 最大迭代次数，默认为 100。
            tol (float): 收敛容差，默认为 1e-6。

        返回:
            tuple: 像素坐标 (u, v)，均为形状为 (N, 1) 的 NumPy 数组。
        """
        # 初始化未畸变的坐标
        x = x_corrected.copy()
        y = y_corrected.copy()

        # 迭代求解未畸变的坐标
        for _ in range(max_iter):
            # 计算半径平方
            r_squared = x**2 + y**2

            # 计算径向畸变
            radial_distortion = 1 + self.k1 * r_squared + self.k2 * r_squared**2

            # 计算切向畸变
            x_tangential = 2 * self.r1 * x * y + self.r2 * (r_squared + 2 * x**2)
            y_tangential = self.r1 * (r_squared + 2 * y**2) + 2 * self.r2 * x * y

            # 计算畸变后的坐标
            x_distorted = x * radial_distortion + x_tangential
            y_distorted = y * radial_distortion + y_tangential

            # 计算误差
            error_x = x_corrected - x_distorted
            error_y = y_corrected - y_distorted

            # 更新未畸变的坐标
            x += error_x
            y += error_y

            # 检查是否收敛
            if np.max(np.abs(error_x)) < tol and np.max(np.abs(error_y)) < tol:
                break

        # 将未畸变的归一化坐标转换为像素坐标
        u = x * self.fx + self.cx
        v = y * self.fy + self.cy

        return u, v

    def interaction_matrix(self, xy_vec, Z):

        xy_matrix = xy_vec.reshape(-1, 2)
        x = xy_matrix[:, 0]
        y = xy_matrix[:, 1]

        Ls = []
        for i in range(len(x)):
            Lsi = np.array([[x[i] / Z, -(1 + x[i] ** 2)],
                            [y[i] / Z, -x[i] * y[i]]])
            Ls.append(Lsi)
        L = np.vstack(Ls)
        return L

    def mpc_controller(self):
        if self.xy_vector is None or self.Z is None:
            return
        
        # 定义视野约束
        def fov_constraints(tau_seq, xy_vector, Z):
            tau_seq = np.asarray(tau_seq)
            tau_seq = tau_seq.flatten()

            c = []
            xy_vector_pred = xy_vector
            for j in range(self.Np):
                Ls = self.interaction_matrix(xy_vector_pred, Z)
                tau = tau_seq[2*j:2*(j+1)].reshape(-1, 1)
                xy_vector_pred = xy_vector_pred + self.Ts * (Ls @ tau)
                u_vec, v_vec = self.transform_xy_to_uv(xy_vector_pred)

                c.extend([u_vec + 0,
                        -u_vec + (self.image_width - 10),
                        v_vec + 0,
                        -v_vec + (self.image_height - 10)])
            c = np.concatenate(c)
            return c.flatten()

        # 定义成本函数
        def cost_function(tau_seq, xy_vector, Z):
            tau_seq = np.asarray(tau_seq)
            tau_seq = tau_seq.flatten()

            J1 = 0
            J2 = 0
            xy_vector_pred = xy_vector
            for i in range(self.Np):
                Ls = self.interaction_matrix(xy_vector_pred, Z)
                tau = tau_seq[2*i:2*(i+1)].reshape(-1, 1)
                xy_vector_pred = xy_vector_pred + self.Ts * (Ls @ tau)
                e = xy_vector_pred - self.desired_xy
                J1 += e.T @ self.Q @ e
                J2 += tau.T @ self.R @ tau
            
            J = J1 + J2
            return J

        # 初始猜测 (2*Np 维向量)
        tau_seq0 = np.tile(self.tau_last, self.Np)

        constraints = [
            # {'type': 'ineq', 'fun': fov_constraints, 'args': (self.xy_vector, self.Z)},
            {'type': 'ineq', 'fun': lambda tau_seq: 0.5 - tau_seq[0::6]},  # vz <= 0.5
            {'type': 'ineq', 'fun': lambda tau_seq: 0.5 + tau_seq[0::6]},  # vz >= -0.5
            {'type': 'ineq', 'fun': lambda tau_seq: 0.3 - tau_seq[1::6]},  # wy <= 0.3
            {'type': 'ineq', 'fun': lambda tau_seq: 0.3 + tau_seq[1::6]},  # wy >= -0.3
        ]

        res = minimize(cost_function, tau_seq0, args=(self.xy_vector, self.Z), method='SLSQP', constraints=constraints, options={'maxiter': 500, 'ftol': 1e-6})

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