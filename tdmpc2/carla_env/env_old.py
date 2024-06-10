import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from collections import deque
from PIL import Image
# from gymnasium.envs import register
# import gymnasium as gym
# from gymnasium import spaces

from gym.envs.registration import register
import gym
from gym import spaces

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


IM_WIDTH = 64
IM_HEIGHT = 64

class CarlaEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    camera_channels=3
    NUM_PIXELS = im_width*im_height*camera_channels

    def __init__(self, render_mode="human"):
        super(CarlaEnv, self).__init__()
        self.render_mode=render_mode

        self.client = None
        self.world = None
        self._communicate()

        self.blueprint_library = self.world.get_blueprint_library()

        self.initial_position=None
        self.obstacle_position=None
        """
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        spawn_location = spawn_point.location
        spawn_location.x = spawn_location.x + 30
        spawn_rotation = spawn_point.rotation

        self.initial_position = spawn_point
        self.obstacle_position = carla.Transform(spawn_location, spawn_rotation)
        """

        #self.initial_position = carla.Transform(carla.Location(x=-56.866,y=140.535,z=0.5999), carla.Rotation(yaw=0.352126))
        #self.obstacle_position = carla.Transform(carla.Location(x=0.5,y=141.0,z=0.5999), carla.Rotation(yaw=0.352126))

        self.camera_data = np.zeros((self.NUM_PIXELS,))
        self.actor_list = []
        self.collision_hist = []

         # Define action space
         # Action space is a 2D vector normalized continuous value between -1 and 1
        self.action_space = spaces.Box(low=np.array([-0.3,-1.0]), high=np.array([1.0, 1.0]), shape=(2,), dtype=np.float64)

        # Define observation space
        #observation_high = np.array([255] * self.NUM_PIXELS + [np.inf, np.inf], dtype=np.float64)
        #observation_low = np.array([0] * self.NUM_PIXELS + [-np.inf, -np.inf], dtype=np.float64)
        # observation_high = np.array([np.inf, np.inf], dtype=np.float64)
        # observation_low = np.array([-np.inf, -np.inf], dtype=np.float64)
        # self.observation_space = spaces.Box(low=observation_low, high=observation_high, dtype=np.float64)

        self.image_space = spaces.Box(low=0, high=255, shape=(self.im_height, self.im_width, self.camera_channels), dtype=np.uint8)
        self.state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64)   # [velocity, s, d]
        self.observation_space = spaces.Dict({
            "image": self.image_space,
            "state": self.state_space
        })

        # Initialize vehicle and camera sensor
        self.vehicle = None
        self.obstacle = None
        self._control = carla.VehicleControl()
        self.camera_sensor = None
        self.collision_sensor = None
        
        self.episode_start = None

    def _init_position(self):
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        spawn_location = spawn_point.location
        spawn_rotation = spawn_point.rotation
        spawn_location.x = spawn_location.x + 30
        spawn_rotation = spawn_point.rotation
        #print(spawn_location.x)

        self.initial_position = spawn_point
        self.obstacle_position = carla.Transform(spawn_location, spawn_rotation)

    # def reset(self, seed=None, options=None):
    #     print("reset")
    #     super().reset(seed=seed, options=options)
    def reset(self):
        if len(self.actor_list):
            self._destroy_actor()
        
        self.collision_hist = []
        self.camera_data = np.zeros((self.NUM_PIXELS,))
        self._control = carla.VehicleControl()

        #print("1")
        self._spawn_vehicle()
        #print("2")
        self._spawn_camera_sensor()
        self._spawn_collision_sensor()

        self.episode_start = time.time()

        observation = self._get_observation()
        return observation

    def render(self, mode="human", **kwargs):
        if mode == "human":
            if self.camera_data is not None:
                image = self.camera_data.reshape((self.im_height, self.im_width, self.camera_channels))
                image = image.astype(np.uint8)
                cv2.imshow("Camera", image)
                cv2.waitKey(1)
        elif mode == "rgb_array":
            # image = self.camera_data.reshape((self.im_height, self.im_width, self.camera_channels))
            if self.camera_data is not None:
                image = self.camera_data.reshape((self.im_height, self.im_width, self.camera_channels))
                image = image.astype(np.uint8)
                cv2.imshow("Camera", image)
                cv2.waitKey(1)
            return image
        elif mode == "depth_array":
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid mode={mode}")
        
    def _apply_action(self, action):
        throttle = float(action[0])
        steer = float(action[1])  
        if throttle > 0:
            self._control.brake = 0.0
            self._control.throttle = throttle
        elif throttle < 0:
            self._control.brake = -throttle
            self._control.throttle = 0.0
        else:
            self._control.brake = 0.0
            self._control.throttle = 0.0
        self._control.steer = steer
        # print(self._control)
        self.vehicle.apply_control(self._control)


    def step(self, action):
        """ Action is a 2d vector normalized continuous value between -1 and 1"""
        self._apply_action(action)
        # throttle = action[0]
        # steer = action[1]
        # if throttle > 0:
        #     self._control.brake = 0.0
        #     self._control.throttle = throttle
        # elif throttle < 0:
        #     self._control.brake = -throttle
        #     self._control.throttle = 0.0
        # else:
        #     self._control.brake = 0.0
        #     self._control.throttle = 0.0
        # self._control.steer = steer
        print(self._control)
        # self.vehicle.apply_control(self._control)
        observation=self._get_observation()
        reward, terminated= self._calculate_reward(observation, action)
        # print("reward: ", reward)
        # print("Action: ", action)
        return observation, reward, terminated, {}

    def _calculate_reward(self, observation, action): 
        terminated = False  
        if self.collision_hist:
            reward = -50
            terminated = True
        # if self.lane_invasion_hist:
        #     reward = -50
        #     terminated = True
        
        velocity = observation["state"][0]
        if velocity > 2 and velocity < 30:
            reward = 100
        elif velocity >= 30:
            reward = 2
        elif velocity < 2:
            reward = -1

        # else:
        #     reward = 1

        return reward, terminated


    def _get_location(self):
        location = self.vehicle.get_location()
        return float(location.x)

    def _communicate(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()

    def _destroy_actor(self):
        #print(len(self.actor_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        #print("destroy")
        self.actor_list=[]

    def _spawn_vehicle(self):
        bp = self.blueprint_library.find('vehicle.tesla.model3')
        while True:
            try:
                self._init_position()
                self.vehicle = self.world.spawn_actor(bp, self.initial_position)
            except RuntimeError as e:
                continue
                #print("retry...")
            self.vehicle.apply_control(self._control)
            self.actor_list.append(self.vehicle)
            return
        
        
        #bp = self.blueprint_library.find('vehicle.toyota.prius')
        #self.obstacle = self.world.spawn_actor(bp, self.obstacle_position)
        #self.obstacle.apply_control(self._control)
        #self.actor_list.append(self.obstacle)
    
    def _spawn_camera_sensor(self):
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', f"{self.im_width}")
        camera_bp.set_attribute('image_size_y', f"{self.im_height}")
        camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.camera_sensor = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera_sensor.listen(lambda data: self._process_img(data))
        self.actor_list.append(self.camera_sensor)
    
    def _process_img(self, image):
        # Process camera image to get observation
        if image is not None:
            # Convert image to numpy array
            image_array = np.frombuffer(image.raw_data, dtype=np.uint8)
            image_array = image_array.reshape((self.im_height, self.im_width, -1))
            image_rgb = image_array[:,:,0:3]
            image_rgb = image_rgb.reshape((self.NUM_PIXELS,))

            # Store the processed image data in self.camera_data
            self.camera_data = image_rgb
        else:
            # If image is None, return zeros array to maintain the shape
            self.camera_data = np.zeros((self.NUM_PIXELS,))

    def _spawn_collision_sensor(self):
        # Define collision sensor blueprint
        collision_bp = self.blueprint_library.find('sensor.other.collision')

        # Spawn collision sensor
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)

        # Register callback for collision events
        self.collision_sensor.listen(lambda event: self._on_collision(event))

        self.actor_list.append(self.collision_sensor)

    def _on_collision(self, event):
        print("YYYYYYYYYYYYY")
        self.collision_hist.append(event)

    def _get_velocity(self):
        velocity = self.vehicle.get_velocity()
        return velocity.length()

    def _get_yaw(self):
        yaw = self.vehicle.get_transform().rotation.yaw
        return yaw

    def _get_observation(self):
        image_observation = self.camera_data
        velocity_observation = self._get_velocity()
        yaw_observation = np.array([self._get_yaw()])
        s, d = 0, 0

        observation = {
            "image": image_observation,
            "state": np.array([velocity_observation, s, d])
        }
        # print("obs type: ", type(observation))

        return observation
    
    def close(self):
        print("close")
        # Clean up resources
        self._destroy_actor()
        
        self.vehicle = None
        self.camera_sensor = None
        self.collision_sensor = None
        self.collision_hist=[]

        time.sleep(1)


if __name__ == "__main__":
    register(
        id="temp-v0",
        entry_point="env:CarlaEnv",
        max_episode_steps=1000000,
    )

    env = gym.make("temp-v0", render_mode="human")
    ob, _ = env.reset()
    print(f"ob_space = {env.observation_space}")
    print(f"ac_space = {env.action_space.shape}")
    env.render(mode="human")
    while True:
        action = env.action_space.sample()
        ob, rew, terminated, info = env.step(action)
        env.render(mode="human")

        # env.render()
        print("reward: ", rew)
        if terminated:
            # print("terminated: ", terminated, " truncated: ", truncated)
            env.reset()
    env.close()

