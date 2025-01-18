#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np

from custom_msgs.msg import DetectionResult
from mpc_ibvs import mvsdk

np.set_printoptions(threshold=np.inf, edgeitems=5, linewidth=200)

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.get_logger().info("Init Start")

        self.detection_pub = self.create_publisher(DetectionResult, '/detection_result', 10)
        self.bridge = CvBridge()

        # 打开相机
        self.DevList = mvsdk.CameraEnumerateDevice()
        self.nDev = len(self.DevList)
        if self.nDev < 1:
            self.get_logger().error("No camera was found!")
            return

        for i, DevInfo in enumerate(self.DevList):
            self.get_logger().info(f"{i}: {DevInfo.GetFriendlyName()} {DevInfo.GetPortType()}")
        i = 0 if self.nDev == 1 else int(input("Select camera: "))
        self.DevInfo = self.DevList[i]
        self.get_logger().info(str(self.DevInfo))
        self.hCamera = 0
        try:
            self.hCamera = mvsdk.CameraInit(self.DevInfo, -1, -1)
        except mvsdk.CameraException as e:
            self.get_logger().error(f"CameraInit Failed({e.error_code}): {e.message}")
            return
        
        self.cap = mvsdk.CameraGetCapability(self.hCamera)
        self.monoCamera = (self.cap.sIspCapacity.bMonoSensor != 0)
        if self.monoCamera:
            mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
        mvsdk.CameraSetTriggerMode(self.hCamera, 0)
        mvsdk.CameraSetAeState(self.hCamera, 0)
        # 曝光时间
        mvsdk.CameraSetExposureTime(self.hCamera, 5 * 1000)
        mvsdk.CameraPlay(self.hCamera)
        self.FrameBufferSize = self.cap.sResolutionRange.iWidthMax * self.cap.sResolutionRange.iHeightMax * (1 if self.monoCamera else 3)
        self.pFrameBuffer = mvsdk.CameraAlignMalloc(self.FrameBufferSize, 16)
        # 设置相机帧率
        self.timer = self.create_timer(1.0 / 60.0, self.timer_callback)

        # detection info
        self.labels = ['A', 'B', 'C', 'D']
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
        
        self.x_coords = self.desired_uv[::2, 0]
        self.y_coords = self.desired_uv[1::2, 0]

        self.last_detected_points = [None, None, None, None]

        self.get_logger().info("MPC detect Init")

    def timer_callback(self):
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera, 200)
            mvsdk.CameraImageProcess(self.hCamera, pRawData, self.pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)

            # 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
            # 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth,
                                   1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))

            frame = cv2.resize(frame, (1280, 1024), interpolation=cv2.INTER_LINEAR)

            self.find_white_circles_using_Contours(frame)
            

        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                self.get_logger().error(f"CameraGetImageBuffer failed({e.error_code}): {e.message}")

    def find_white_circles_using_Contours(self, img):
        # 应用阈值化
        _, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

        # 找到二值图像中的轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        centers = []

        for i in range(min(4, len(contours))):
            contour = contours[i]
            
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append((cX, cY))
        
        if centers:
            mean_center = np.mean(centers, axis=0)
            d_u, d_v = int(mean_center[0]), int(mean_center[1])

            if 1 :
                Z = 1

            # if self.depth_image is not None:
            #     self.get_logger().info("depth")
            #     Z = self.depth_image[d_v, d_u]
            #     if np.isnan(Z) or Z <= 0:
            #         self.get_logger().warn("Invalid depth value at pixel (%d, %d), skipping this frame." % (d_u, d_v))
            #         return
                    
                s, identified_points= self.identify_key_points(centers)
                s = s.flatten().astype(float)
                # print(identified_points)

                # Publish detection result
                detection_result = DetectionResult()
                detection_result.s = s.tolist()
                detection_result.z = float(Z)
                self.detection_pub.publish(detection_result)

                for i, identified_point in enumerate(identified_points):
                    cv2.circle(img, identified_point[0], 1, (0, 255, 0), -1)
                    cv2.putText(img, self.labels[i], (identified_point[0][0] + 10, identified_point[0][1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                for i, (x, y) in enumerate(zip(self.x_coords, self.y_coords)):
                    cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
                    cv2.putText(img, self.labels[i], (x + 10, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                cv2.imshow("Detected Red Dots", img)
                cv2.waitKey(1)

    def identify_key_points(self, centers):
        # Sort centers to determine A, B, C, D based on x and y coordinates
        A = sorted(centers, key=lambda x: x[0])[:2]
        C = sorted(centers, key=lambda x: x[0], reverse=True)[:2]
        B = sorted(centers, key=lambda x: x[1])[:2]
        D = sorted(centers, key=lambda x: x[1], reverse=True)[:2]

        # Find the intersection points
        point1 = list(set(A) & set(B))
        point4 = list(set(C) & set(D))
        point3 = list(set(A) & set(D))
        point2 = list(set(B) & set(C))

        # If there are no intersections, use the last detected points
        if len(point1) == 0:
            point1 = self.last_detected_points[0] if self.last_detected_points[0] is not None else [(0, 0)]
        if len(point4) == 0:
            point4 = self.last_detected_points[3] if self.last_detected_points[3] is not None else [(0, 0)]
        if len(point3) == 0:
            point3 = self.last_detected_points[2] if self.last_detected_points[2] is not None else [(0, 0)]
        if len(point2) == 0:
            point2 = self.last_detected_points[1] if self.last_detected_points[1] is not None else [(0, 0)]

        # Update the last detected points
        self.last_detected_points = [point1, point2, point3, point4]

        # Prepare the result in the requested format as a numpy array
        points = []

        # Add points to the array
        points.append([float(point1[0][0])])  # x1
        points.append([float(point1[0][1])])  # y1
        points.append([float(point2[0][0])])  # x2
        points.append([float(point2[0][1])])  # y2
        points.append([float(point3[0][0])])  # x3
        points.append([float(point3[0][1])])  # y3
        points.append([float(point4[0][0])])  # x4
        points.append([float(point4[0][1])])  # y4

        # Return the points as a numpy array and the identified points
        return np.array(points, dtype=float), [point1, point2, point3, point4]


def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
