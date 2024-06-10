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

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')
from PIL.PngImagePlugin import PngImageFile, PngInfo
try:
    import queue
except ImportError:
    import Queue as queue
IM_WIDTH = 64
IM_HEIGHT = 64
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import atexit
import signal
import sys

# Other imports...

def cleanup():
    print('Cleaning up...')
    if 'env' in globals():
        env.close()

def handle_signal(sig, frame):
    print(f'Signal {sig} received. Cleaning up...')
    cleanup()
    sys.exit(0)

# Register the cleanup function to be called on normal program exit
atexit.register(cleanup)

# Register signal handlers for abrupt terminations
signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

        self.start()

    def start(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue(1000)
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)

    def tick(self, timeout):
        self.frame = self.world.tick()
        logger.debug("World ticked, frame: %s", self.frame)
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        # while True:
        #     data = sensor_queue.get(timeout=timeout)
        #     if data.frame == self.frame:
        #         return data
        while True:
            try:
                data = sensor_queue.get(timeout=timeout)
                if data.frame == self.frame:
                    logger.debug("Data retrieved for frame: %s", data.frame)
                    return data
            except queue.Empty:
                logger.error("Failed to retrieve data: queue is empty, increase timeout")
                data = sensor_queue.get(timeout=timeout+10)
                if data.frame == self.frame:
                    logger.debug("Data retrieved for frame: %s", data.frame)
                    return data

                # raise

def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    # print("draw_image array shape", array.shape)
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    # print("draw_image image_surface shape", image_surface.get_size())
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))

def draw_image_obs(surface, image, blend=False):
    # Assuming image is in the shape (252, 84, 3) or similar
    # if image.ndim == 3 and image.shape[0] in (84, 252) and image.shape[1] in (84, 252) and image.shape[2] == 3:
        # Transpose the image back to (height, width, channels)
    # image = image.transpose((1, 2, 0))
    # print("image shape", image.shape)

    # Convert the image array to a Pygame surface
    image_surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
    # print("image_surface shape", image_surface.get_size())

    if blend:
        image_surface.set_alpha(100)
    
    surface.blit(image_surface, (380, 0))
    # pygame.display.update(pygame.Rect(380, 0, 800-image.shape[0], image.shape[1]))

    
def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))


class Sun(object):
    def __init__(self, azimuth, altitude):
        self.azimuth = azimuth
        self.altitude = altitude
        self._t = 0.0

    def tick(self, delta_seconds):
        self._t += 0.008 * delta_seconds
        self._t %= 2.0 * math.pi
        self.azimuth += 0.25 * delta_seconds
        self.azimuth %= 360.0
        # self.altitude = (70 * math.sin(self._t)) - 20  # [50, -90]
        min_alt, max_alt = [20, 90]
        self.altitude = 0.5 * (max_alt + min_alt) + 0.5 * (max_alt - min_alt) * math.cos(self._t)

    def __str__(self):
        return 'Sun(alt: %.2f, azm: %.2f)' % (self.altitude, self.azimuth)


class Storm(object):
    def __init__(self, precipitation):
        self._t = precipitation if precipitation > 0.0 else -50.0
        self._increasing = True
        self.clouds = 0.0
        self.rain = 0.0
        self.wetness = 0.0
        self.puddles = 0.0
        self.wind = 0.0
        self.fog = 0.0

    def tick(self, delta_seconds):
        delta = (1.3 if self._increasing else -1.3) * delta_seconds
        self._t = clamp(delta + self._t, -250.0, 100.0)
        self.clouds = clamp(self._t + 40.0, 0.0, 60.0)
        self.rain = clamp(self._t, 0.0, 80.0)
        self.wind = 5.0 if self.clouds <= 20 else 90 if self.clouds >= 70 else 40
        if self._t == -250.0:
            self._increasing = True
        if self._t == 100.0:
            self._increasing = False

    def __str__(self):
        return 'Storm(clouds=%d%%, rain=%d%%, wind=%d%%)' % (self.clouds, self.rain, self.wind)


class Weather(object):
    def __init__(self, world, changing_weather_speed):
        self.world = world
        self.reset()
        self.weather = world.get_weather()
        self.changing_weather_speed = changing_weather_speed
        self._sun = Sun(self.weather.sun_azimuth_angle, self.weather.sun_altitude_angle)
        self._storm = Storm(self.weather.precipitation)

    def reset(self):
        weather_params = carla.WeatherParameters(sun_altitude_angle=90.)
        self.world.set_weather(weather_params)

    def tick(self):
        self._sun.tick(self.changing_weather_speed)
        self._storm.tick(self.changing_weather_speed)
        self.weather.cloudiness = self._storm.clouds
        self.weather.precipitation = self._storm.rain
        self.weather.precipitation_deposits = self._storm.puddles
        self.weather.wind_intensity = self._storm.wind
        self.weather.fog_density = self._storm.fog
        self.weather.wetness = self._storm.wetness
        self.weather.sun_azimuth_angle = self._sun.azimuth
        self.weather.sun_altitude_angle = self._sun.altitude
        self.world.set_weather(self.weather)

    def __str__(self):
        return '%s %s' % (self._sun, self._storm)

class CarlaEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    camera_channels=3
    NUM_PIXELS = im_width*im_height*camera_channels

    def __init__(self,
                 render_mode="human",  # "human", "rgb_array", "depth_array"
                 render_display=1,  # 0, 1
                 record_display_images=0,  # 0, 1
                 record_rl_images=0,  # 0, 1
                 changing_weather_speed=0.0,  # [0, +inf)
                 display_text=1,  # 0, 1
                 rl_image_size=84,
                 max_episode_steps=1000,
                 frame_skip=2,
                 is_other_cars=True,
                 start_lane=None,
                 fov=60,  # degrees for rl camera
                 num_cameras=5,
                 port=2000
                 ):
        
        print("CarlaEnv init")
        if record_display_images:
            assert render_display
        self.render_display = render_display
        self.save_display_images = record_display_images
        self.save_rl_images = record_rl_images
        self.changing_weather_speed = changing_weather_speed
        self.display_text = display_text
        self.rl_image_size = rl_image_size
        self._max_episode_steps = max_episode_steps  # DMC uses this
        self.frame_skip = frame_skip
        self.is_other_cars = is_other_cars
        self.start_lane = start_lane
        self.num_cameras = num_cameras
        self.port = port
        self.render_obs = 1
        self.fov = fov  
        self.actor_list = []

        self.setup()

        # dummy variables given bisim's assumption on deep-mind-control suite APIs
        self.action_space = spaces.Box(low=np.array([-1.0,-1.0]), high=np.array([1.0, 1.0]), shape=(2,), dtype=np.float64)
        
        num_frames = 1
        self.image_space = gym.spaces.Box(
			low=0, high=255, shape=(rl_image_size, rl_image_size*num_cameras, num_frames*3), dtype=np.uint8
		)
        self._frames = deque([], maxlen=num_frames)
        # [velocity, s_progress, d, alpha, throttle, brake, steer, prev_throttle, prev_brake, prev_steer]
        self.state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64)   
        self.observation_space = spaces.Dict({
            "image": self.image_space,
            "state": self.state_space
        })
        # self.observation_space = self.image_space
        self.reward_range = None
        self.metadata = None

        # roaming carla agent
        # self.agent = None
        self.count = 0
        self.dist_s = 0
        self.return_ = 0
        self.velocities = []
        
        self.steer = 0
        self.throttle = 0
        self.brake = 0

        self.speed = 0
        self.s = 0
        self.d = 0
        self.alpha = 0
        self.waypoint_list = []
        self.f2c = None
        self.x_spline = None
        self.y_spline = None
        self.path_length = 0
        self.invasion_count = 0
        self.vel_s = 0
        self.reward = 0
        # self.collision_intensities_during_last_time_step = []
        # self.invasions_during_last_time_step = []

        self.world.tick()


        # self.reset()  # creates self.agent


    def setup(self):

        if self.render_display:
            pygame.init()
            self.display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.font = get_font()
            self.clock = pygame.time.Clock()

        # if self.render_obs:
        #     self.display_obs = pygame.display.set_mode((400, 200), pygame.HWSURFACE | pygame.DOUBLEBUF)
        #     self.font_obs = get_font()
        #     self.clock_obs = pygame.time.Clock()

        self.client = carla.Client('localhost', self.port)
        self.client.set_timeout(15.0)

        self.world = self.client.load_world("Town01")
        self.map = self.world.get_map()
        print("Map name:", self.map.name)
        assert self.map.name == "Carla/Maps/Town01"
        self.map_resolution = 2.0  # fot conversion fails over 2.0
        self.global_planner = GlobalRoutePlanner(self.map, self.map_resolution)

        # remove old vehicles and sensors (in case they survived)
        self.world.tick()
        actor_list = self.world.get_actors()
        for vehicle in actor_list.filter("*vehicle*"):
            # if vehicle.id != self.vehicle.id:
            print("Warning: removing old vehicle")
            vehicle.destroy()
        for sensor in actor_list.filter("*sensor*"):
            print("Warning: removing old sensor")
            sensor.destroy()

        self.vehicle = None
        self.vehicle_start_pose = None
        self.vehicles_list = []  # their ids
        self.vehicles = None
        self.reset_vehicle()  # creates self.vehicle
        self.actor_list.append(self.vehicle)

        blueprint_library = self.world.get_blueprint_library()

        if self.render_display:
            self.camera_rgb = self.world.spawn_actor(
                blueprint_library.find('sensor.camera.rgb'),
                carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
                # carla.Transform(carla.Location(x=1.2, z=1.5), carla.Rotation(pitch=-15)),
                attach_to=self.vehicle)
            self.actor_list.append(self.camera_rgb)

        # we'll use up to five cameras, which we'll stitch together
        # bp = blueprint_library.find('sensor.camera.rgb')
        # bp = blueprint_library.find('sensor.camera.depth')
        bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        bp.set_attribute('image_size_x', str(self.rl_image_size))
        bp.set_attribute('image_size_y', str(self.rl_image_size))
        bp.set_attribute('fov', str(self.fov))
        location = carla.Location(x=1.6, z=1.7)
        self.camera_rl = self.world.spawn_actor(bp, carla.Transform(location, carla.Rotation(yaw=0.0)), attach_to=self.vehicle)
        self.camera_rl_left = self.world.spawn_actor(bp, carla.Transform(location, carla.Rotation(yaw=-float(self.fov))), attach_to=self.vehicle)
        self.camera_rl_lefter = self.world.spawn_actor(bp, carla.Transform(location, carla.Rotation(yaw=-2*float(self.fov))), attach_to=self.vehicle)
        self.camera_rl_right = self.world.spawn_actor(bp, carla.Transform(location, carla.Rotation(yaw=float(self.fov))), attach_to=self.vehicle)
        self.camera_rl_righter = self.world.spawn_actor(bp, carla.Transform(location, carla.Rotation(yaw=2*float(self.fov))), attach_to=self.vehicle)
        self.actor_list.append(self.camera_rl)
        self.actor_list.append(self.camera_rl_left)
        self.actor_list.append(self.camera_rl_lefter)
        self.actor_list.append(self.camera_rl_right)
        self.actor_list.append(self.camera_rl_righter)

        bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        self.actor_list.append(self.collision_sensor)
        self._collision_intensities_during_last_time_step = []

        # setup lane invasion sensor
        bp = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.lane_invasion_sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
        self.lane_invasion_sensor.listen(lambda event: self._on_invasion(event))
        self.actor_list.append(self.lane_invasion_sensor)
        self._invasions_during_last_time_step = []

        if self.save_display_images or self.save_rl_images:
            import datetime
            now = datetime.datetime.now()
            image_dir = "images-" + now.strftime("%Y-%m-%d-%H-%M-%S")
            os.mkdir(image_dir)
            self.image_dir = image_dir

        if self.render_display:
            self.sync_mode = CarlaSyncMode(self.world, self.camera_rgb, self.camera_rl, self.camera_rl_left, self.camera_rl_lefter, self.camera_rl_right, self.camera_rl_righter, fps=20)
        else:
            self.sync_mode = CarlaSyncMode(self.world, self.camera_rl, self.camera_rl_left, self.camera_rl_lefter, self.camera_rl_right, self.camera_rl_righter, fps=20)

        # weather
        self.weather = Weather(self.world, self.changing_weather_speed)

        self.reset_other_vehicles()

    def _on_invasion(self, event):
        self._invasions_during_last_time_step = []
        for marking in event.crossed_lane_markings:
            print(f"marking info: type: {marking.type} color: {marking.color}, lane change: {marking.lane_change}, width: {marking.width}")
            if marking.type == carla.LaneMarkingType.NONE:
                print("None type marking")
                self._invasions_during_last_time_step.append(marking)
        print("lane invasion", event.crossed_lane_markings)
        self.invasion_count += 1


    def _on_collision(self, event):
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        print('Collision (intensity {})'.format(intensity))
        self._collision_intensities_during_last_time_step.append(intensity)


    def _get_transform(self):
        transform = self.vehicle.get_transform()
        x, y, yaw = transform.location.x, transform.location.y, transform.rotation.yaw
        return x,y,yaw

    def calculate_distance(init_pos, goal_pos):
        distance = math.sqrt((init_pos.location.x - goal_pos.location.x)**2 + (init_pos.location.y - goal_pos.location.y)**2)
    
    def draw_global_plan(self, global_plan):
        self.waypoint_list = []
        for waypoint, road_option in global_plan:
            self.waypoint_list.append(waypoint)
        draw_waypoints(self.world, self.waypoint_list)


    def convert_global_to_frenet(self, waypoint_list):
        try:
            wpts_list = []
            for waypoint in waypoint_list:
                # print(waypoint)
                wpts_list.append([waypoint.transform.location.x, waypoint.transform.location.y])
            self.f2c = FrenetCartesianConverter(wpts_list)
            x_spline = self.f2c.x_spline
            y_spline = self.f2c.y_spline

            return x_spline, y_spline
        except Exception as e:
            print("Error converting global to frenet", e)
            return None, None
    
    def reset(self):
        self.reset_vehicle()
        self.world.tick()
        # self.reset_other_vehicles()
        # self.world.tick()
        self.count = 0
        self.dist_s = 0
        self.return_ = 0
        self.reward = 0
        self.velocities = []

        # for _ in range(self._frames.maxlen):
        #     obs = self._get_observation()
        # return obs
        # # get obs:
        obs, _, _, _ = self.step(action=None)
        # self.reward = reward
        return obs
    
    def reset_vehicle(self):
        self.initial_position = random.choice(self.map.get_spawn_points())
        if self.vehicle is None:
            # create vehicle
            blueprint_library = self.world.get_blueprint_library()
            vehicle_blueprint = blueprint_library.find('vehicle.audi.a2')
            self.vehicle = self.world.spawn_actor(vehicle_blueprint, self.initial_position)
        else:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            self.vehicle.set_target_velocity(carla.Vector3D())
            self.vehicle.set_transform(self.initial_position)

        self.vehicle.set_target_velocity(carla.Vector3D())
        # self.vehicle.set_angular_velocity(carla.Vector3D())
    
        goal_position = random.choice(self.map.get_spawn_points())
        while is_within_distance(self.initial_position, goal_position, 200):
            print("goal too close to initial position, resampling")
            goal_position = random.choice(self.map.get_spawn_points())  
        self.goal_position = goal_position   
        self.global_plan = self.global_planner.trace_route(self.initial_position.location, self.goal_position.location)

        self.waypoint_list = []
        for waypoint, road_option in self.global_plan:
            self.waypoint_list.append(waypoint)
        self.x_spline, self.y_spline = self.convert_global_to_frenet(self.waypoint_list)
        if self.x_spline is None or self.y_spline is None:
            print("Error: x_spline or y_spline is None")
            self.reset_vehicle()
            print("Resetting vehicle check if control returned here")
            return
        else:
            self.path_length = self.f2c.path_length

        self.draw_global_plan(self.global_plan)

        


    def reset_other_vehicles(self):
        if not self.is_other_cars:
            return

        # clear out old vehicles
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
        self.world.tick()
        self.vehicles_list = []

        traffic_manager = self.client.get_trafficmanager()  
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        traffic_manager.set_synchronous_mode(True)
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]

        num_vehicles = 30
        other_car_transforms = []
        for _ in range(num_vehicles):
            lane_id = random.choice([1, 2, 3, 4])
            start_x = 1.5 + 3.5 * lane_id
            start_y = random.uniform(-40., 40.)
            transform = carla.Transform(carla.Location(x=start_x, y=start_y, z=0.1), carla.Rotation(yaw=-90))
            other_car_transforms.append(transform)

        # Spawn vehicles
        batch = []
        for n, transform in enumerate(other_car_transforms):
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(carla.command.SpawnActor(blueprint, transform).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True)))
        for response in self.client.apply_batch_sync(batch, False):
            self.vehicles_list.append(response.actor_id)

        for response in self.client.apply_batch_sync(batch):
            if response.error:
                pass
            else:
                self.vehicles_list.append(response.actor_id)
    
    def get_state_obs(self):
        state = [float(self.speed), float(self.s - self.prev_s), float(self.d), float(self.alpha), float(self.throttle), 
                 float(self.brake), float(self.steer), float(self.prev_throttle), float(self.prev_brake), float(self.prev_steer)]
        return state
    
    def step(self, action):
        rewards = []
        for _ in range(self.frame_skip):  # default 1
            #print(action)
            next_obs, reward, done, info = self._simulator_step(action)
            rewards.append(reward)
            if done:
                break
        next_obs = {'image': next_obs}
        return next_obs, np.mean(rewards), done, info  # just last info?


    def _simulator_step(self, action, dt=0.05):
        self.dt = dt
        done = False
        x, y, yaw = self._get_transform()
        yaw_rad = math.radians(yaw)
        # print("yaw: ", yaw)
        self.prev_s, self.prev_d, self.prev_alpha = self.s, self.d, self.alpha
        frenet_pose = self.f2c.get_frenet([x, y, yaw_rad])
        s = frenet_pose[0]
        d = frenet_pose[1]
        alpha_rad = frenet_pose[2]
        alpha = math.degrees(alpha_rad)
        # alpha = 180-alpha 
        self.s, self.d, self.alpha = s, d, alpha
        # print("s: ", s, " d: ", d, " alpha: ", alpha)

        if self.render_display:
            if should_quit():
                return
            self.clock.tick()

        if action is not None:
            steer = float(action[0])
            throttle_brake = float(action[1])
            if throttle_brake >= 0.0:
                throttle = throttle_brake
                brake = 0.0
            else:
                throttle = 0.0
                brake = -throttle_brake

            assert 0.0 <= throttle <= 1.0
            assert -1.0 <= steer <= 1.0
            assert 0.0 <= brake <= 1.0

            steer = steer * 0.5  # 0.5 max steer

            # clip the rate of change of controls
            if self.prev_steer is not None:
                steer = self.prev_steer + np.clip(steer - self.prev_steer, -0.1, 0.1)
            if self.prev_throttle is not None:
                throttle = self.prev_throttle + np.clip(throttle - self.prev_throttle, -0.3, 0.3)
            if self.prev_brake is not None:
                brake = self.prev_brake + np.clip(brake - self.prev_brake, -0.5, 0.5)

        

            vehicle_control = carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake,
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False
            )
            self.vehicle.apply_control(vehicle_control)
        else:
            throttle, steer, brake = 0., 0., 0.

        self.prev_steer = self.steer
        self.prev_throttle = self.throttle
        self.prev_brake = self.brake

        self.throttle = throttle
        self.steer = steer
        self.brake = brake

        info = {}
        info['reason_episode_ended'] = ''
        # dist_from_center, vel_s, speed, done, info = self.dist_from_center_lane(self.vehicle, info)
        collision_intensities_during_last_time_step = sum(self._collision_intensities_during_last_time_step)
        assert collision_intensities_during_last_time_step >= 0.
        # collision_cost = 0.0001 * collision_intensities_during_last_time_step

        self.speed = 3.6 * math.sqrt(self.vehicle.get_velocity().x**2 + self.vehicle.get_velocity().y**2 + self.vehicle.get_velocity().z**2)
        self.vel_s = self.speed * np.cos(self.alpha)


        reward, done = self._calculate_reward(info)
        # reward = self.vel_s * self.dt - collision_cost - abs(self.steer)  # doesn't work if 0.001 cost collisions

        info['crash_intensity'] = collision_intensities_during_last_time_step
        info['steer'] = self.steer
        info['brake'] = self.brake
        info['distance'] = self.vel_s * dt

        self.dist_s += self.vel_s * dt
        self.return_ += reward
        self.reward = reward
        self.collision_intensities_during_last_time_step = collision_intensities_during_last_time_step
        self.weather.tick()


        self.count += 1
        next_obs = self.render()
        # done = False
        if self.count >= self._max_episode_steps:
            print("Episode success: I've reached the episode horizon ({}).".format(self._max_episode_steps))
            info['reason_episode_ended'] = 'success'
            done = True


        return next_obs, reward, done, info

    def _calculate_reward(self, info): 

        done = False
        reward = 0.0

        if self.speed > 1:
            reward += 1 + 2.0 * (self.s - self.prev_s) + 0.1 * self.throttle - 0.1 * self.brake - 0.01 * abs(self.steer) - 0.01 * abs(self.d) - 0.01 * abs(self.alpha) 
        collision_intensities_during_last_time_step = sum(self._collision_intensities_during_last_time_step)
        # collision_cost = 0.0001 * collision_intensities_during_last_time_step
        # reward = reward * (1 - self.steer)
        if self.speed < 0.02 and self.count >= 100 and self.count % 100 == 0:  # a hack, instead of a counter
            reward = -0.3
            # reward -= 5
            print("Episode fail: speed too small ({}), think I'm stuck! (frame {})".format(self.speed, self.count))
            info['reason_episode_ended'] = 'stuck'
            done = True
        if collision_intensities_during_last_time_step > 0.0:
            # reward -= 500
            # reward -= 10
            reward = -10 - 0.0001 * collision_intensities_during_last_time_step - 0.1 * self.throttle
            print("Episode fail: collision intensity too high ({}), I've crashed! (frame {})".format(collision_intensities_during_last_time_step, self.count))
            info['reason_episode_ended'] = 'crash'
            done = True
            self._collision_intensities_during_last_time_step.clear()  # clear it ready for next time step

            # self.collision_intensities_during_last_time_step = 0.0

        if self._invasions_during_last_time_step: 
            print("Warning: LANE INVASION")
            reward = -0.5 + 0.1 * self.throttle
            # reward -= 100
            # if self.invasion_count == 5:
            self.invasion_count = 0

            print("Episode fail: lane invasion, I've crossed the line! (frame {})".format(self.count))
            info['reason_episode_ended'] = 'lane invasion'
            done = True
            self._invasions_during_last_time_step.clear()

        if abs(self.d) > 6.0:
            reward = -0.5 + 0.1 * self.throttle
            # reward -= 450
            print("Episode fail: d too large ({}), I've went the other way! (frame {})".format(self.d, self.count))
            info['reason_episode_ended'] = 'd too large'
            done = True
        
        if self.s > self.path_length:
            reward += 1000
            print("Episode success: I've reached the end of the path! (frame {})".format(self.count))
            info['reason_episode_ended'] = 'success'
            done = True

        return reward, done

    # def _calculate_reward(self, info): 

    #     done = False
    #     reward = 0.0

    #     # velocity = observation["state"][0]

    #     if self.s <= 0.0:
    #         reward -= 0.01


    #     # print("path length: ", self.path_length)
    #     # Progress reward
    #     # if self.speed > 5:
    #     #     reward += (self.s / self.path_length) * 10

    #     # # Lane keeping reward
    #     # if abs(d) > 3.0:
    #     #     reward += (3 - abs(d)) * 0.1

    #     # penalty for braking 
    #     if self.brake > 0:
    #         reward -= 0.01
        
    #     #penalty for large delta change in action
    #     if abs(self.steer - self.prev_steer) > 0.1:
    #         reward -= 0.5 * abs(self.steer - self.prev_steer)
    #     if abs(self.throttle - self.prev_throttle) > 0.2:
    #         reward -= 0.1 * abs(self.throttle - self.prev_throttle)
    #     if abs(self.brake - self.prev_brake) > 0.2:
    #         reward -= 0.1 * abs(self.brake - self.prev_brake)

    #     reward += (self.s - self.prev_s) * 5.0

    #     # reward to slow speed when steering
        
    #     # orientation reward
    #     if abs(self.alpha) < 20 and self.speed > 5:
    #         reward += 0.1
    #     else:
    #         reward -= abs(self.alpha) * 0.01

    #     reward -= abs(self.d) * 0.01

    #     collision_intensities_during_last_time_step = sum(self._collision_intensities_during_last_time_step)
    #     collision_cost = 0.0001 * collision_intensities_during_last_time_step

    #     # reward += self.vel_s * self.dt - collision_cost - abs(self.steer)  # doesn't work if 0.001 cost collisions
    #     reward -= collision_cost  # doesn't work if 0.001 cost collisions

    #     # reward += self.speed/(abs(self.steer)*5.0 + 1.0) * 0.1

    #     # reward -= self.speed * abs(self.steer) * 0.01
        
    #     if self.speed < 0.02 and self.count >= 100 and self.count % 100 == 0:  # a hack, instead of a counter
    #         # reward -= 0.3
    #         # reward -= 5
    #         print("Episode fail: speed too small ({}), think I'm stuck! (frame {})".format(self.speed, self.count))
    #         info['reason_episode_ended'] = 'stuck'
    #         done = True
    #     if collision_intensities_during_last_time_step > 0.0:
    #         # reward -= 500
    #         # reward -= 10
    #         print("Episode fail: collision intensity too high ({}), I've crashed! (frame {})".format(collision_intensities_during_last_time_step, self.count))
    #         info['reason_episode_ended'] = 'crash'
    #         done = True
    #         self._collision_intensities_during_last_time_step.clear()  # clear it ready for next time step

    #         # self.collision_intensities_during_last_time_step = 0.0

    #     if self._invasions_during_last_time_step:
    #         # reward -= 0.5
    #         # reward -= 100
    #         print("Episode fail: lane invasion, I've crossed the line! (frame {})".format(self.count))
    #         info['reason_episode_ended'] = 'lane invasion'
    #         done = True
    #         self._invasions_during_last_time_step.clear()

    #     if abs(self.d) > 5.0:
    #         # reward -= 0.5
    #         # reward -= 450
    #         print("Episode fail: d too large ({}), I've went the other way! (frame {})".format(self.d, self.count))
    #         info['reason_episode_ended'] = 'd too large'
    #         done = True
        
    #     if self.s > self.path_length:
    #         reward += 1000
    #         print("Episode success: I've reached the end of the path! (frame {})".format(self.count))
    #         info['reason_episode_ended'] = 'success'
    #         done = True

    #     return reward, done
            
    def render(self, mode="human", **kwargs):
        self.draw_global_plan(self.global_plan)

        # if self.render_display:
        # if mode == "human":
        snapshot, image_rgb, image_rl, image_rl_left, image_rl_lefter, image_rl_right, image_rl_righter = self.sync_mode.tick(timeout=2.0)
        
        ## convert to CityScapesPalette
        image_rl.convert(carla.ColorConverter.CityScapesPalette)
        image_rl_left.convert(carla.ColorConverter.CityScapesPalette)
        image_rl_lefter.convert(carla.ColorConverter.CityScapesPalette)
        image_rl_right.convert(carla.ColorConverter.CityScapesPalette)
        image_rl_righter.convert(carla.ColorConverter.CityScapesPalette)
        ##

        ## convert depth image
        # image_rl.convert(carla.ColorConverter.Depth)
        # image_rl_left.convert(carla.ColorConverter.Depth)
        # image_rl_lefter.convert(carla.ColorConverter.Depth)
        # image_rl_right.convert(carla.ColorConverter.Depth)
        # image_rl_righter.convert(carla.ColorConverter.Depth)
        ##
        
        draw_image(self.display, image_rgb)
        if self.display_text:
            self.display.blit(self.font.render('frame %d' % self.count, True, (255, 255, 255)), (8, 10))
            self.display.blit(self.font.render('highway progression %4.1f m/s (%5.1f m) (%5.2f speed)' % (self.vel_s, self.dist_s, self.speed), True, (255, 255, 255)), (8, 28))
            self.display.blit(self.font.render('reward %5.2f (return %.2f)' % (self.reward, self.return_), True, (255, 255, 255)), (8, 64))
            self.display.blit(self.font.render('collision intensity %5.2f ' % self.collision_intensities_during_last_time_step, True, (255, 255, 255)), (8, 82))
            self.display.blit(self.font.render('thottle %5.2f,steer %3.2f, brake %3.2f' % (self.throttle, self.steer, self.brake), True, (255, 255, 255)), (8, 100))
            self.display.blit(self.font.render(str(self.weather), True, (255, 255, 255)), (8, 118))
            self.display.blit(self.font.render('s: %4.1f, d: %4.1f, alpha: %4.1f, path_length: %4.1f' % (self.s, self.d, self.alpha, self.path_length), True, (255, 255, 255)), (8, 136))


        # elif mode == "rgb_array":
        #     snapshot, image_rl, image_rl_left, image_rl_lefter, image_rl_right, image_rl_righter = self.sync_mode.tick(timeout=2.0)


        rgbs = []
        if self.num_cameras == 1:
            ims = [image_rl]
        elif self.num_cameras == 3:
            ims = [image_rl_left, image_rl, image_rl_right]
        elif self.num_cameras == 5:
            ims = [image_rl_lefter, image_rl_left, image_rl, image_rl_right, image_rl_righter]
        else:
            raise ValueError("num cameras must be 1 or 3 or 5")
        for im in ims:
            bgra = np.array(im.raw_data).reshape(self.rl_image_size, self.rl_image_size, 4)  # BGRA format
            # bgra = np.array(im).reshape(self.rl_image_size, self.rl_image_size, 4)  # BGRA format
            bgr = bgra[:, :, :3]  # BGR format (84 x 84 x 3)
            rgb = np.flip(bgr, axis=2)  # RGB format (84 x 84 x 3)
            rgbs.append(rgb)
        rgb = np.concatenate(rgbs, axis=1)  # (84 x 252 x 3)
        # rgb = rgb.transpose((2, 0, 1))  # (252 x 84 x 3)

        if mode == "human" and self.save_display_images:
            image_name = os.path.join(self.image_dir, "display%08d.jpg" % self.count)
            pygame.image.save(self.display, image_name)
            # ffmpeg -r 20 -pattern_type glob -i 'display*.jpg' carla.mp4
        if self.save_rl_images:
            image_name = os.path.join(self.image_dir, "rl%08d.png" % self.count)
            im = Image.fromarray(rgb)
            metadata = PngInfo()
            metadata.add_text("throttle", str(self.throttle))
            metadata.add_text("steer", str(self.steer))
            metadata.add_text("brake", str(self.brake))
            im.save(image_name, "PNG", pnginfo=metadata)


        next_obs = rgb  # (84 x 252 x 3) or (84 x 420 x 3)
        # print("next_obs shape: ", next_obs.shape)
        # print("self.observation_space shape: ", self.observation_space.shape)
        assert next_obs.shape == self.observation_space.shape
        draw_image_obs(self.display, next_obs)
        pygame.display.flip()
    
        return next_obs

        


    def set_spectator(self):
        spectator = self.world.get_spectator()
        transform = self.vehicle.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))



    def close(self):
        print('destroying actors.')
        for actor in self.actor_list:
            actor.destroy()
        print('\ndestroying %d vehicles' % len(self.vehicles_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
        time.sleep(0.5)
        pygame.quit()
        print('done.')

if __name__ == "__main__":
    register(
        id="temp-v0",
        entry_point="env_iso:CarlaEnv",
        max_episode_steps=1000000,
    )

    env = gym.make("temp-v0", render_mode="human")
    print("env: ", env)
    ob, reward = env.reset()
    print(f"ob_space = {env.observation_space}")
    print(f"ac_space = {env.action_space.shape}")
    # env.render(mode="human")
    step = 0
    while True:
        print("ISO")
        action = env.action_space.sample()
        ob, rew, terminated, info = env.step(action)
        # env.render(mode="human")

        # env.render()
        print("reward: ", rew)
        if terminated:
            print("terminated: ", terminated)
            # print("terminated: ", terminated, " truncated: ", truncated)
            ob, reward = env.reset()
        elif step % 100 == 0:
            print("step reset: ", step)
            ob,reward = env.reset()
        # elif step > 1000:
        #     # env.close()
        #     break
        
        step += 1
    env.close()

