#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tf
import tf.transformations
import cv2
import time
import rospy
import torch
import numpy as np
import message_filters

from PIL import Image as PILImage
from functools import partial
from cv_bridge import CvBridge
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformations import euler_from_quaternion, quaternion_from_euler, quaternion_from_matrix

from std_msgs.msg import String, Int16MultiArray
from sensor_msgs.msg import Image, JointState
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import PoseStamped

# processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
# vla = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b",  attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16,  low_cpu_mem_usage=True,  trust_remote_code=True).to("cuda:0")

# image: Image.Image = torch.rand(3, 256, 256)
# image = Image.fromarray((image * 255).byte().permute(1, 2, 0).numpy())
# prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"

# inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
# action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

# create class for the foundation model runner
class foundation_model_runner:
    def __init__(self):
        self.im_size = 256
        self.num_timesteps = 1000

        # set up the model:
        self.processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b",  attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16,  low_cpu_mem_usage=True,  trust_remote_code=True).to("cuda:0")

        self.bridge = CvBridge()
        self.pub = rospy.Publisher('/cartesian_impedance_controller/equilibrium_pose', PoseStamped, queue_size=10)

    def image_callback(self, image_data, robot_sub):
        cv_image = self.bridge.imgmsg_to_cv2(image_data, "bgr8")
        image = PILImage.fromarray(cv_image).resize((self.im_size, self.im_size), PILImage.LANCZOS)

        # Predict Action (7-DoF; un-normalize for BridgeData V2)
        inputs = self.processor(self.goal_instruction, image).to("cuda:0", dtype=torch.bfloat16)
        action = self.vla.predict_action(**inputs, unnorm_key="", do_sample=False)

        print("--------------------")
        print("Time: ", self.t, "\n")
        print("Goal Instruction: ", self.goal_instruction, "\n")
        print("Action: ", action, "\n")

        # add action to the robot state:
        action[0] += robot_sub.O_T_EE[12]
        action[1] += robot_sub.O_T_EE[13]
        action[2] += robot_sub.O_T_EE[14]

        # franka robot upright is 0, 0, 0, 1 but the output of the robot model is flipped so that the gripper is facing up (incorrectly) so we need to flip the quaternion
        # action[3] = -action[3]
        # action[4] = -action[4]
        # action[5] = -action[5]

        robot_quat = quaternion_from_matrix(np.transpose(np.reshape(robot_sub.O_T_EE, (4, 4))))

        if self.t != 0:
            action_message = PoseStamped()
            action_message.header.stamp = rospy.Time.now()
            action_message.header.frame_id = "0"
            # action_message.header.frame_id = "base_link"

            action_message.pose.position.x = action[0]
            action_message.pose.position.y = action[1]
            action_message.pose.position.z = action[2]
            quaternions = quaternion_from_euler(action[3], action[4], action[5])
            action_message.pose.orientation.x = 1 # quaternions[0] + robot_quat[0]
            action_message.pose.orientation.y = 0 # quaternions[1] + robot_quat[1]
            action_message.pose.orientation.z = 0 # quaternions[2] + robot_quat[2]
            action_message.pose.orientation.w = 0 # quaternions[3] + robot_quat[3]
            self.pub.publish(action_message)

            print("Action: ", action_message, "\n")

        self.t+=1
        if self.t > self.num_timesteps:
            print("Stopping the foundation model - at the end of the timestep limit.")
            self.image_subscriber.unregister()

    def run_model(self):
        rospy.init_node('realsense_camera_node', anonymous=True)
        rate = rospy.Rate(100)

        self.t = 0
        instruction = input("Please enter the natural language instruction:")
        self.goal_instruction = instruction

        input("Press [Enter] to start.")

        # self.image_subscriber = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.robot_sub = message_filters.Subscriber('/franka_state_controller/franka_states', FrankaState)
        self.image_subscriber = message_filters.Subscriber('/camera/color/image_raw', Image)
        subscribers = [self.image_subscriber, self.robot_sub]
        self.ts_sub = message_filters.ApproximateTimeSynchronizer(subscribers, queue_size=50, slop=100.0, allow_headerless=True)
        self.ts_sub.registerCallback(self.image_callback)
        
        time.sleep(1.0)

        print("finshed registering the callback.")
        while not rospy.is_shutdown() and self.t < self.num_timesteps:
            rate.sleep()
        print("Model run complete.")

if __name__ == '__main__':
    runner = foundation_model_runner()
    runner.run_model()