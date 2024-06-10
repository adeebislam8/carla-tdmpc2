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
from gym.envs.registration import register
import gym
from gym import spaces
import torch

# try:
#     sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass
import carla
from PythonAPI.carla.agents.navigation.global_route_planner import GlobalRoutePlanner
from PythonAPI.carla.agents.tools.misc import draw_waypoints, is_within_distance, distance_vehicle
from frenet_cartesian_converter import FrenetCartesianConverter

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

    def __init__(self, render_mode="human", num_frames=3, render_size=64):
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
        self.prev_action = [0,0]
         # Define action space
         # Action space is a 2D vector normalized continuous value between -1 and 1
        self.action_space = spaces.Box(low=np.array([-1.0,-1.0]), high=np.array([1.0, 1.0]), shape=(2,), dtype=np.float64)

        # Define observation space
        #observation_high = np.array([255] * self.NUM_PIXELS + [np.inf, np.inf], dtype=np.float64)
        #observation_low = np.array([0] * self.NUM_PIXELS + [-np.inf, -np.inf], dtype=np.float64)
        # observation_high = np.array([np.inf, np.inf], dtype=np.float64)
        # observation_low = np.array([-np.inf, -np.inf], dtype=np.float64)
        # self.observation_space = spaces.Box(low=observation_low, high=observation_high, dtype=np.float64)

        # self.image_space = spaces.Box(low=0, high=255, shape=(self.camera_channels, self.im_height, self.im_width), dtype=np.uint8)
        self.image_space = gym.spaces.Box(
			low=0, high=255, shape=(num_frames*3, render_size, render_size), dtype=np.uint8
		)
        self._frames = deque([], maxlen=num_frames)
        self.state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)   # [velocity, s, d, alpha]
        self.observation_space = spaces.Dict({
            "image": self.image_space,
            "state": self.state_space
        })
        self._render_size = render_size

        # Initialize vehicle and camera sensor
        self.vehicle = None
        self.obstacle = None
        self._control = carla.VehicleControl()
        self.camera_sensor = None
        self.collision_sensor = None
        
        self.episode_start = None

        self.map = self.world.get_map()
        self.sampling_resolution = 2.0
        self.global_planner = GlobalRoutePlanner(self.map, self.sampling_resolution)
        self.global_plan = None

        self.s = 0
        self.d = 0
        self.alpha = 0


    def _init_position(self):
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        # spawn_location = spawn_point.location
        # spawn_rotation = spawn_point.rotation
        # spawn_location.x = spawn_location.x 
        # spawn_rotation = spawn_point.rotation
        #print(spawn_location.x)

        self.initial_position = spawn_point
        # self.obstacle_position = carla.Transform(spawn_location, spawn_rotation)

    def calculate_distance(init_pos, goal_pos):
        distance = math.sqrt((init_pos.location.x - goal_pos.location.x)**2 + (init_pos.location.y - goal_pos.location.y)**2)
    
    def draw_global_plan(self, global_plan):
        self.waypoint_list = []
        for waypoint, road_option in global_plan:
            self.waypoint_list.append(waypoint)
        draw_waypoints(self.world, self.waypoint_list)


    def convert_global_to_frenet(self, waypoint_list):
        wpts_list = []
        for waypoint in waypoint_list:
            # print(waypoint)
            wpts_list.append([waypoint.transform.location.x, waypoint.transform.location.y])
        self.f2c = FrenetCartesianConverter(wpts_list)
        x_spline = self.f2c.x_spline
        y_spline = self.f2c.y_spline

        return x_spline, y_spline
    
    def reset(self):
        print("reset")
        # super().reset(seed=seed, options=options)


        self.camera_data = np.zeros((self.NUM_PIXELS,))
        self._control = carla.VehicleControl()

        if len(self.actor_list) == 0:
            # self._destroy_actor()
            # print("destroy actor")
            print("1")
            self._spawn_vehicle()
            print("spawned vehicle")
            self._spawn_camera_sensor()
            print("spawned camera sensor")

            self._spawn_collision_sensor()
            print("spawn collision sensor")
        else:
            self.vehicle.apply_control(self._control)
            self.vehicle.set_target_velocity(carla.Vector3D(x=0, y=0, z=0))
            self.vehicle.set_target_angular_velocity(carla.Vector3D(x=0, y=0, z=0))
            self._init_position()
            self.vehicle.set_transform(self.initial_position)
            #time out
            time.sleep(1)
        self.collision_hist = []

        # if len(self.actor_list) > 0:
        #     self._destroy_actor()
        #     print("destroy actor")
        # self.actor_list = []
        # self.collision_hist = []
        # self.camera_data = np.zeros((self.NUM_PIXELS,))
        # self._control = carla.VehicleControl()

        # print("1")
        # self._spawn_vehicle()
        # print("spawned vehicle")
        # self._spawn_camera_sensor()
        # print("spawned camera sensor")

        # self._spawn_collision_sensor()
        # print("spawn collision sensor")
        goal_position = random.choice(self.map.get_spawn_points())
        # distance_to_goal = self.calculate_distance(self.initial_position, goal_position)
        # while distance_to_goal < 100:
        while is_within_distance(self.initial_position, goal_position, 100):
            goal_position = random.choice(self.map.get_spawn_points())  
        self.goal_position = goal_position   

        # print("start_pos: ", self.initial_position)
        # print("goal_pos: ", self.goal_position) 

        self.global_plan = self.global_planner.trace_route(self.initial_position.location, self.goal_position.location)
        self.waypoint_list = []
        for waypoint, road_option in self.global_plan:
            self.waypoint_list.append(waypoint)
        self.x_spline, self.y_spline = self.convert_global_to_frenet(self.waypoint_list)
        self.path_length = self.f2c.path_length
        # print(self.global_plan)
        # self.draw_global_plan(self.global_plan)
        self.episode_start = time.time()
        for _ in range(self._frames.maxlen):
            obs = self._get_observation()
			# print("obs type", type(obs))
        return obs


        observation = self._get_observation()
        return observation


    def step(self, action):
        """ Action is a 2d vector normalized continuous value between -1 and 1"""
        self._apply_action(action)
        observation=self._get_observation()
        reward, terminated= self._calculate_reward(observation, action)
        print("env.py reward: ", reward)
        print("env.py Action: ", action)
        self.prev_action = action
        return observation, reward, terminated, {}

    def _calculate_reward(self, observation, action): 

        terminated = False
        reward = 0.0

        # Check for collision
        if self.collision_hist:
            reward -= 50
            terminated = True
            return reward, terminated

        # Extract necessary information
        velocity = observation["state"][0]
        x, y, yaw = self._get_transform()
        frenet_pose = self.f2c.get_frenet([x, y, yaw])
        s = frenet_pose[0]
        d = frenet_pose[1]
        alpha = frenet_pose[2]
        self.s, self.d, self.alpha = s, d, alpha

        # Progress reward
        reward += (s / self.path_length) * 1.0

        # Lane keeping reward
        if abs(d) > 3.0:
            reward += (3 - abs(d)) * 0.1

        # penalty for braking 
        if action[0] < 0:
            reward -= 0.1
        
        #penalty for large delta change in action
        if abs(action[1] - self.prev_action[1]) > 0.5:
            reward -= 0.1
        if abs(action[0] - self.prev_action[0]) > 0.5:
            reward -= 0.1

        # Speed reward
        if 10 <= velocity <= 20:
            reward += velocity * 0.1
        else:
            reward -= abs(velocity - 15) * 0.01

        # orientation reward
        if abs(alpha) < 0.4:
            reward += 0.1
        else:
            reward -= abs(alpha) * 0.01

        # Time penalty
        
        # Check if goal is reached
        # if self.calculate_distance(self.vehicle.get_transform(), self.goal_position) < 5:
        #     reward += 100
        #     terminated = True

        return reward, terminated
    def set_spectator(self):
        spectator = self.world.get_spectator()
        transform = self.vehicle.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))

    def render(self, mode="human", **kwargs):
        if mode == "human":
            if self.camera_data is not None:
                image = self.camera_data.reshape((self.im_height, self.im_width, self.camera_channels))
                image = image.astype(np.uint8)
                self.draw_global_plan(self.global_plan)
                self.set_spectator()
                cv2.imshow("Camera", image)
                cv2.waitKey(1)
                return image
        elif mode == "rgb_array":
            # image = self.camera_data.reshape((self.im_height, self.im_width, self.camera_channels))
            if self.camera_data is not None:
                image = self.camera_data.reshape((self.im_height, self.im_width, self.camera_channels))
                image = image.astype(np.uint8)
                self.draw_global_plan(self.global_plan)
                self.set_spectator()

                cv2.imshow("Camera", image)
                cv2.waitKey(1)
                print("image shape: ", image.shape)
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



    def _get_transform(self):
        transform = self.vehicle.get_transform()
        x, y, yaw = transform.location.x, transform.location.y, transform.rotation.yaw
        return x,y,yaw

    def _get_location(self):
        location = self.vehicle.get_location()
        # x, y, yaw = location.x, location.y
        return float(location.x)

    def _communicate(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.03
        self.world.apply_settings(settings)


    def _destroy_actor(self):
        print(len(self.actor_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        print("destroy")
        self.actor_list=[]

    def _spawn_vehicle(self):
        bp = self.blueprint_library.find('vehicle.tesla.model3')
        while True:
            try:
                self._init_position()
                print("initial_position: ", self.initial_position)
                self.vehicle = self.world.spawn_actor(bp, self.initial_position)
                self.actor_list.append(self.vehicle)
                print("spawning vehicle...")
            except RuntimeError as e:
                print("runtime error...")

                continue
            self.vehicle.apply_control(self._control)
            return
        
        
        #bp = self.blueprint_library.find('vehicle.toyota.prius')
        #self.obstacle = self.world.spawn_actor(bp, self.obstacle_position)
        #self.obstacle.apply_control(self._control)
        #self.actor_list.append(self.obstacle)
    
    def _spawn_camera_sensor(self):
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', f"{self.im_width}")
        camera_bp.set_attribute('image_size_y', f"{self.im_height}")
        camera_transform = carla.Transform(carla.Location(x=1.8, z=1.0))
        try:
            self.camera_sensor = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
            self.camera_sensor.listen(lambda data: self._process_img(data))
            self.actor_list.append(self.camera_sensor)
    
        except RuntimeError as e:
            print("runtime error...")
            self._destroy_actor()
            self._spawn_vehicle()
            self._spawn_camera_sensor()
            return
    
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
        try:
            self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
            # Register callback for collision events
            self.collision_sensor.listen(lambda event: self._on_collision(event))

            self.actor_list.append(self.collision_sensor)
        except RuntimeError as e:
            print("runtime error...")
            self._destroy_actor()
            self._spawn_vehicle()
            self._spawn_camera_sensor()
            self._spawn_collision_sensor()
            return


    def _on_collision(self, event):
        # print("YYYYYYYYYYYYY")
        self.collision_hist.append(event)

    def _get_velocity(self):
        velocity = self.vehicle.get_velocity()
        return velocity.length()

    def _get_yaw(self):
        yaw = self.vehicle.get_transform().rotation.yaw
        return yaw

    def _get_observation(self):
        # image_observation = self.camera_data.reshape((self.camera_channels, self.im_height, self.im_width))
        frame = self.render(
            mode='rgb_array', width=self._render_size, height=self._render_size
        ).transpose(2, 0, 1)
        # print("frame type", type(frame))
        self._frames.append(frame)
        # return torch.from_numpy(np.concatenate(self._frames))
        # frames = np.concatenate(self._frames)
        # print("Frame shape: ", frames.shape)
        # print("Frame type: ", type(frames))
        image_observation = torch.from_numpy(np.concatenate(self._frames))
        # print("image_obs type: ", type(image_observation))
        # print("camera_data shape: ", self.camera_data.shape)
        print("image_observation shape: ", image_observation.shape)



        velocity_observation = self._get_velocity()
        # yaw_observation = np.array([self._get_yaw()])
        # s, d, alpha = 0, 0, 0

        observation = {
            "image": image_observation,
            "state": np.array([velocity_observation, self.s, self.d, self.alpha])
        }
        print("obs_state: ", observation["state"])

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
            print("terminated: ", terminated)
            # print("terminated: ", terminated, " truncated: ", truncated)
            env.reset()
    env.close()

