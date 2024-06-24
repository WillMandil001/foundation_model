#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyrealsense2 as rs
import rospy
import cv2

import numpy as np

from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def main():
    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


    pipeline.start(config)

    # rospy.sleep(2)


    rospy.init_node('realsense_camera_node', anonymous=True)
    img_pub = rospy.Publisher('/camera/color/image_raw', Image, queue_size=1)
    depth_pub = rospy.Publisher('/camera/color/depth_raw', Image, queue_size=1)
    bridge = CvBridge()

    crop_height = 124
    crop = 0

    while not rospy.is_shutdown():
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # color_image = color_image[0:500, 100:100+500]
        # depth_image = depth_image[0:500, 100:100+500]

        # resized_color_image = cv2.resize(np.array(color_image), (640, 480), interpolation=cv2.INTER_AREA)
        # resized_depth_image = cv2.resize(np.array(depth_image), (640, 480), interpolation=cv2.INTER_AREA)

        resized_color_image = cv2.resize(np.array(color_image), (256, 256), interpolation=cv2.INTER_AREA)
        resized_depth_image = cv2.resize(np.array(depth_image), (256, 256), interpolation=cv2.INTER_AREA)


        ros_color_image = bridge.cv2_to_imgmsg(np.array(resized_color_image), "bgr8")
        ros_depth_image = bridge.cv2_to_imgmsg(np.array(resized_depth_image), "passthrough")

        ros_color_image.header.stamp = rospy.Time.now()
        ros_depth_image.header.stamp = rospy.Time.now()
        img_pub.publish(ros_color_image)
        depth_pub.publish(ros_depth_image)


if __name__ == '__main__':
    main()