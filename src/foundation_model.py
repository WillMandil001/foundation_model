#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import jax
import time
import rospy
import numpy as np
import jax.numpy as jnp
import message_filters

from functools import partial
from cv_bridge import CvBridge
# from xela_server.msg import XStream
from transformations import euler_from_quaternion, quaternion_from_euler
from std_msgs.msg import String, Int16MultiArray
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, JointState

from octo.utils.spec import ModuleSpec
from octo.utils.jax_utils import initialize_compilation_cache
from octo.model.octo_model import OctoModel
from octo.utils.train_utils import (merge_params, process_text)
from octo.model.components.tokenizers import LowdimObsTokenizer
from octo.model.components.action_heads import L1ActionHead


# create class for the foundation model runner
class foundation_model_runner:
    def __init__(self):
        self.im_size = 256
        self.num_timesteps = 100
        self.action_dim = 7
        self.testing_wihtout_model = False
        self.batch_size = 1
        self.horizon = 2

        # set up the model:
        pretrained_path = "hf://rail-berkeley/octo-base"
        # initialize_compilation_cache()
        # pretrained_model = OctoModel.load_pretrained(pretrained_path)
        self.model = OctoModel.load_pretrained(pretrained_path)
        # print(pretrained_model.get_pretty_spec())
        # config = pretrained_model.config
        # del config["model"]["observation_tokenizers"]["wrist"]
        # text_processor = pretrained_model.text_processor
        # text = "pick up the ball and throw it to the big red dog, then slap the ball with the bat."
        # batch = {"task": {"language_instruction": [text.encode('utf-8')]},
        #         "observation": {"image_primary": np.random.uniform(0, 256, size=(self.batch_size, self.horizon, self.im_size, self.im_size, 3)).astype(np.int8),
        #                         # "proprio": np.random.uniform(-2.0, 2.0, size=(self.batch_size, self.horizon, self.action_dim)).astype(np.float32),
        #                         "pad_mask": np.ones((self.batch_size, self.horizon)).astype(np.float32)}}
        # example_batch = process_text(batch, text_processor)
        # # # config["model"]["observation_tokenizers"]["proprio"] = ModuleSpec.create(LowdimObsTokenizer, n_bins=256, bin_type="normal", low=-2.0, high=2.0, obs_keys=["proprio"])
        # config["model"]["heads"]["action"] = ModuleSpec.create(L1ActionHead, pred_horizon=self.horizon, action_dim=self.action_dim, readout_key="readout_action")
        # model = OctoModel.from_config(config, example_batch, text_processor, verbose=True)
        # merged_params = merge_params(model.params, pretrained_model.params)
        # self.model = model.replace(params=merged_params)
        # print(self.model.get_pretty_spec())
        # del pretrained_model

        self.goal_instruction = ""
        self.action_list = []
        self.goal_image = jnp.zeros((self.im_size, self.im_size, 3), dtype=np.uint8)
        self.t = 0
        self.bridge = CvBridge()
        self.show_image = True
        self.STEP_DURATION = 0.2
        self.pub = rospy.Publisher('/cartesian_impedance_example_controller/equilibrium_pose', PoseStamped, queue_size=10)
        self.last_tstep = time.time()
        self.CONTEXT_WINDOW_SIZE = 2
        self.images = [np.zeros((self.im_size, self.im_size, 3), dtype=np.uint8) for i in range(self.CONTEXT_WINDOW_SIZE+1)]
        self.robot_states = [np.array([0.3, 0.0, 0.15, 0.001*self.t, 0.0, 0.0, 0.0]) for i in range(self.CONTEXT_WINDOW_SIZE+1)]
        print("Foundation model runner initialized.")

    def image_callback(self, image_data, robot_joint_data):#, image_data):
        if time.time() > self.last_tstep + self.STEP_DURATION:
            self.last_tstep = time.time()

            cv_image = self.bridge.imgmsg_to_cv2(image_data, "bgr8")

            self.images.append(cv_image)
            self.robot_states.append(np.array(robot_joint_data.position[:7])) # dummy robot state to be filled from robot later

            pad_mask = np.ones((self.horizon)).astype(np.float32)
            input_images = np.stack(self.images[len(self.images) - self.CONTEXT_WINDOW_SIZE:])
            input_robot_states = np.stack(self.robot_states[len(self.images) - self.CONTEXT_WINDOW_SIZE:])

            observations = {"image_primary": input_images.astype(np.int8),                    # "image": np.random.uniform(0, 256, size=(horizon, 256, 256, 3)).astype(np.int8),
                            # "proprio": input_robot_states.astype(np.float32),       # "proprio": np.random.uniform(-2.0, 2.0, size=(horizon, 14)).astype(np.float32),
                            "pad_mask": np.ones((self.horizon)).astype(np.float32)}   # "pad_mask": np.ones((horizon)).astype(np.float32)
            print(observations["image_primary"].shape, observations["pad_mask"].shape)

            if self.t != 0:

                actions = self.model.sample_actions(jax.tree_map(lambda x: x[None], observations), self.task, rng=jax.random.PRNGKey(0))  # [0]

                # Unnormalize
                # actions = (actions[0] * self.model.dataset_statistics["action"]["std"] + self.model.dataset_statistics["action"]["mean"])

                self.action_list.append(actions[0])
                print("robot joint data: \n {} \n `image shape: \n {} \n robot state shape: \n{} \n pad mask shape: \n {} \n actions: \n {}".format(robot_joint_data, input_images.shape, input_robot_states.shape, pad_mask.shape, actions))
                print("--------------")

                action_message = PoseStamped()
                action_message.header.stamp = rospy.Time.now()
                action_message.header.frame_id = "base_link"
                action_message.pose.position.x = actions[0][0][0]
                action_message.pose.position.y = actions[0][0][1]
                action_message.pose.position.z = actions[0][0][2]

                # need to convert the euler angles to quaternion
                quaternions = quaternion_from_euler(actions[0][0][3], actions[0][0][4], actions[0][0][5])
                action_message.pose.orientation.x = quaternions[0]
                action_message.pose.orientation.y = quaternions[1]
                action_message.pose.orientation.z = quaternions[2]
                action_message.pose.orientation.w = quaternions[3]

                self.pub.publish(action_message)  # publish the action to the robot using ros publisher

            self.t+=1
            if self.t > self.num_timesteps:
                print("Stopping the foundation model - at the end of the timestep limit.")
                self.image_subscriber.unregister()

    def run_model(self):
        rospy.init_node('realsense_camera_node', anonymous=True)
        rate = rospy.Rate(100)
        
        instruction = input("Please enter the natural language instruction:")
        self.task = self.model.create_tasks(texts=[instruction])
        self.goal_instruction = instruction
        print(self.goal_instruction)

        input("Press [Enter] to start.")

        self.last_tstep = time.time()
        self.images = []
        self.goals = []
        self.robot_states = []
        self.t = 0
        time.sleep(1.0)

        # self.xela_sub = message_filters.Subscriber('/xServTopic', XStream)
        self.robot_sub = message_filters.Subscriber('/joint_states', JointState)
        self.image_subscriber = message_filters.Subscriber('/camera/color/image_raw', Image)
        subscribers = [self.image_subscriber, self.robot_sub]  # self.xela_sub
        self.ts_sub = message_filters.ApproximateTimeSynchronizer(subscribers, queue_size=50, slop=100.0, allow_headerless=True)
        self.ts_sub.registerCallback(self.image_callback)
        time.sleep(1.0)

        print("finshed registering the callback.")
        while not rospy.is_shutdown() and self.t < self.num_timesteps:
            rate.sleep()
        print("Model run complete.")

        # plot the action list one plot for each joint:
        import matplotlib.pyplot as plt
        ACTION_DIM_LABELS = ['x', 'y', 'z', 'yaw', 'pitch', 'roll', 'grasp']
        for i in range(self.action_dim):
            plt.plot([a[0][i] for a in self.action_list], label=ACTION_DIM_LABELS[i])
        plt.legend()
        plt.show()


if __name__ == '__main__':
    runner = foundation_model_runner()
    runner.run_model()