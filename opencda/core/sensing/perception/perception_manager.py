0# -*- coding: utf-8 -*-
"""
Perception module base.
"""

# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib
import pyqtgraph.opengl as gl
# from PyQt5 import QtWidgets, uic, QtGui, QtCore
import weakref
import sys
import time
import math
import carla
import cv2
import numpy as np
import open3d as o3d

import opencda.core.sensing.perception.sensor_transformation as st
from opencda.core.common.misc import \
    cal_distance_angle, get_speed, get_speed_sumo
from opencda.core.sensing.perception.obstacle_vehicle import \
    ObstacleVehicle
from opencda.core.sensing.perception.static_obstacle import \
    StaticObstacle, FenceObstacle
from opencda.core.sensing.perception.static_obstacle import TrafficLight
from opencda.core.sensing.perception.o3d_lidar_libs import \
    o3d_visualizer_init, o3d_pointcloud_encode, o3d_visualizer_show, \
    o3d_camera_lidar_fusion, o3d_visualizer_with_color,o3d_pointcloud_encode_color, azimuthangle
from opencda.scenariomanager.carla_data_provider import CarlaDataProvider

VEHICLE_SEGMENTATION_COLOR = (142, 0, 0, 255)
UNLABELED_SEGMENTATION_COLOR = (0, 0, 0, 255)
FENCE_SEGMENTATION_COLOR = (40, 40,100,255)

blueprints = {
    'vehicle.audi.a2':6,
    'walker.pedestrian.0002':1,
    'vehicle.mercedes-benz.coupe':3,
    'vehicle.tesla.model3':2,
    'vehicle.chevrolet.impala':4,
    'vehicle.jeep.wrangler_rubicon':5}

class CameraSensor:
    """
    Camera manager for vehicle or infrastructure.

    Parameters
    ----------
    vehicle : carla.Vehicle
        The carla.Vehicle, this is for cav.

    world : carla.World
        The carla world object, this is for rsu.

    global_position : list
        Global position of the infrastructure, [x, y, z]

    relative_position : str
        Indicates the sensor is a front or rear camera. option:
        front, left, right.

    Attributes
    ----------
    image : np.ndarray
        Current received rgb image.
    sensor : carla.sensor
        The carla sensor that mounts at the vehicle.

    """

    def __init__(self, vehicle, world, relative_position, global_position):
        if vehicle is not None:
            world = vehicle.get_world()

        blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
        blueprint.set_attribute('fov', '100')

        spawn_point = self.spawn_point_estimation(relative_position,
                                                  global_position)

        if vehicle is not None:
            self.sensor = world.spawn_actor(
                blueprint, spawn_point, attach_to=vehicle)
        else:
            self.sensor = world.spawn_actor(blueprint, spawn_point)

        self.image = None
        self.timstamp = None
        self.frame = 0
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CameraSensor._on_rgb_image_event(
                weak_self, event))

        # camera attributes
        self.image_width = int(self.sensor.attributes['image_size_x'])
        self.image_height = int(self.sensor.attributes['image_size_y'])

    @staticmethod
    def spawn_point_estimation(relative_position, global_position):

        pitch = 0
        carla_location = carla.Location(x=0, y=0, z=0)

        if global_position is not None:
            carla_location = carla.Location(
                x=global_position[0],
                y=global_position[1],
                z=global_position[2])
            pitch = -35

        if relative_position == 'front':
            carla_location = carla.Location(x=carla_location.x + 2.5,
                                            y=carla_location.y,
                                            z=carla_location.z + 1.0)
            yaw = 0

        elif relative_position == 'right':
            carla_location = carla.Location(x=carla_location.x + 0.0,
                                            y=carla_location.y + 0.3,
                                            z=carla_location.z + 1.8)
            yaw = 100

        elif relative_position == 'left':
            carla_location = carla.Location(x=carla_location.x + 0.0,
                                            y=carla_location.y - 0.3,
                                            z=carla_location.z + 1.8)
            yaw = -100
        else:
            carla_location = carla.Location(x=carla_location.x - 2.0,
                                            y=carla_location.y,
                                            z=carla_location.z + 1.5)
            yaw = 180

        carla_rotation = carla.Rotation(roll=0, yaw=yaw, pitch=pitch)
        spawn_point = carla.Transform(carla_location, carla_rotation)

        return spawn_point

    @staticmethod
    def _on_rgb_image_event(weak_self, event):
        """CAMERA  method"""
        self = weak_self()
        if not self:
            return
        image = np.array(event.raw_data)
        image = image.reshape((self.image_height, self.image_width, 4))
        # we need to remove the alpha channel
        image = image[:, :, :3]

        self.image = image
        self.frame = event.frame
        self.timestamp = event.timestamp


# class Se
class LidarSensor:
    """
    Lidar sensor manager.

    Parameters
    ----------
    vehicle : carla.Vehicle
        The carla.Vehicle, this is for cav.

    world : carla.World
        The carla world object, this is for rsu.

    config_yaml : dict
        Configuration dictionary for lidar.

    global_position : list
        Global position of the infrastructure, [x, y, z]

    Attributes
    ----------
    o3d_pointcloud : 03d object
        Received point cloud, saved in o3d.Pointcloud format.

    sensor : carla.sensor
        Lidar sensor that will be attached to the vehicle.

    """

    def __init__(self, vehicle, world, config_yaml, global_position):
        if vehicle is not None:
            world = vehicle.get_world()
        blueprint = world.get_blueprint_library().find('sensor.lidar.ray_cast')

        # set attribute based on the configuration
        blueprint.set_attribute('upper_fov', str(config_yaml['upper_fov']))
        blueprint.set_attribute('lower_fov', str(config_yaml['lower_fov']))
        blueprint.set_attribute('channels', str(config_yaml['channels']))
        blueprint.set_attribute('range', str(config_yaml['range']))
        blueprint.set_attribute(
            'points_per_second', str(
                config_yaml['points_per_second']))
        blueprint.set_attribute(
            'rotation_frequency', str(
                config_yaml['rotation_frequency']))
        blueprint.set_attribute(
            'dropoff_general_rate', str(
                config_yaml['dropoff_general_rate']))
        blueprint.set_attribute(
            'dropoff_intensity_limit', str(
                config_yaml['dropoff_intensity_limit']))
        blueprint.set_attribute(
            'dropoff_zero_intensity', str(
                config_yaml['dropoff_zero_intensity']))
        blueprint.set_attribute(
            'noise_stddev', str(
                config_yaml['noise_stddev']))

        # spawn sensor
        if global_position is None:
            spawn_point = carla.Transform(carla.Location(x=-0.5, z=1.9))
        else:
            spawn_point = carla.Transform(carla.Location(x=global_position[0],
                                                         y=global_position[1],
                                                         z=global_position[2]))
        if vehicle is not None:
            self.sensor = world.spawn_actor(
                blueprint, spawn_point, attach_to=vehicle)
        else:
            self.sensor = world.spawn_actor(blueprint, spawn_point)

        # lidar data
        self.data = None
        self.timestamp = None
        self.frame = 0
        # open3d point cloud object
        self.o3d_pointcloud = o3d.geometry.PointCloud()

        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: LidarSensor._on_data_event(
                weak_self, event))


    @staticmethod
    def _on_data_event(weak_self, event):
        """Lidar  method"""
        self = weak_self()
        if not self:
            return

        # retrieve the raw lidar data and reshape to (N, 4)
        data = np.copy(np.frombuffer(event.raw_data, dtype=np.dtype('f4')))
        # (x, y, z, intensity)
        data = np.reshape(data, (int(data.shape[0] / 4), 4))

        self.data = data
        self.frame = event.frame
        self.timestamp = event.timestamp


class DepthCameraSensor:
    def __init__(self, vehicle, world, config_yaml, global_position):
        if vehicle is not None:
            world = vehicle.get_world()

        blueprint = \
            world.get_blueprint_library(). \
                find('sensor.camera.depth')

        blueprint.set_attribute('fov', '100')

        # spawn sensor
        if global_position is None:
            spawn_point = carla.Transform(carla.Location(x=-0.5, z=1.9))
        else:
            spawn_point = carla.Transform(carla.Location(x=global_position[0],
                                                         y=global_position[1],
                                                         z=global_position[2]))

        if vehicle is not None:
            self.sensor = world.spawn_actor(
                blueprint, spawn_point, attach_to=vehicle)
        else:
            self.sensor = world.spawn_actor(blueprint, spawn_point)

        self.timestamp = None
        self.frame = 0
        self.data = None

        self.image_width = int(self.sensor.attributes['image_size_x'])
        self.image_height = int(self.sensor.attributes['image_size_y'])

        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: DepthCameraSensor._on_data_event(
                weak_self, event))

    @staticmethod
    def _on_data_event(weak_self, event):
        """Semantic Lidar  method"""
        self = weak_self()
        if not self:
            return

        event.convert(carla.ColorConverter.LogarithmicDepth)
        array = np.frombuffer(event.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (event.height, event.width, 4))
        array = array[:, :, :3]
        # cv2.imshow("", array)
        # cv2.waitKey(2)
        # print(array)
        self.data = np.reshape(event.raw_data, newshape=(self.image_height, self.image_width, 4))
        self.frame = event.frame
        self.timestamp = event.timestamp


class SemanticCameraSensor:
    def __init__(self, vehicle, world, config_yaml, global_position):
        if vehicle is not None:
            world = vehicle.get_world()

        blueprint = \
            world.get_blueprint_library(). \
                find('sensor.camera.semantic_segmentation')

        blueprint.set_attribute('fov', '100')
        # spawn sensor
        if global_position is None:
            spawn_point = carla.Transform(carla.Location(x=-0.5, z=1.9))
        else:
            spawn_point = carla.Transform(carla.Location(x=global_position[0],
                                                         y=global_position[1],
                                                         z=global_position[2]))
        if vehicle is not None:
            self.sensor = world.spawn_actor(
                blueprint, spawn_point, attach_to=vehicle)
        else:
            self.sensor = world.spawn_actor(blueprint, spawn_point)

        self.timestamp = None
        self.frame = 0
        self.data = None

        self.image_width = int(self.sensor.attributes['image_size_x'])
        self.image_height = int(self.sensor.attributes['image_size_y'])

        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: SemanticCameraSensor._on_data_event(
                weak_self, event))

    @staticmethod
    def _on_data_event(weak_self, event):
        """Semantic Lidar  method"""
        self = weak_self()
        if not self:
            return
        event.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(event.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (event.height, event.width, 4))
        array = array[:, :, :3]
        # cv2.imshow("", array)
        # cv2.waitKey(2)

        self.data = np.reshape(event.raw_data, newshape=(self.image_height, self.image_width, 4))
        self.frame = event.frame
        self.timestamp = event.timestamp


class SemanticLidarSensor:
    """
    Semantic lidar sensor manager. This class is used when data dumping
    is needed.

    Parameters
    ----------
    vehicle : carla.Vehicle
        The carla.Vehicle, this is for cav.

    world : carla.World
        The carla world object, this is for rsu.

    config_yaml : dict
        Configuration dictionary for lidar.

    global_position : list
        Global position of the infrastructure, [x, y, z]

    Attributes
    ----------
    o3d_pointcloud : 03d object
        Received point cloud, saved in o3d.Pointcloud format.

    sensor : carla.sensor
        Lidar sensor that will be attached to the vehicle.


    """

    def __init__(self, vehicle, world, config_yaml, global_position):
        if vehicle is not None:
            world = vehicle.get_world()

        blueprint = \
            world.get_blueprint_library(). \
                find('sensor.lidar.ray_cast_semantic')

        # set attribute based on the configuration
        blueprint.set_attribute('upper_fov', str(config_yaml['upper_fov']))
        blueprint.set_attribute('lower_fov', str(config_yaml['lower_fov']))
        blueprint.set_attribute('channels', str(config_yaml['channels']))
        blueprint.set_attribute('range', str(config_yaml['range']))
        # blueprint.set_attribute('horizontal_fov','360')
        blueprint.set_attribute(
            'points_per_second', str(
                config_yaml['points_per_second']))
        blueprint.set_attribute(
            'rotation_frequency', str(
                config_yaml['rotation_frequency']))

        # spawn sensor
        if global_position is None:
            spawn_point = carla.Transform(carla.Location(x=-0.5, z=1.9))
        else:
            spawn_point = carla.Transform(carla.Location(x=global_position[0],
                                                         y=global_position[1],
                                                         z=global_position[2]))

        if vehicle is not None:
            self.sensor = world.spawn_actor(
                blueprint, spawn_point, attach_to=vehicle)
        else:
            self.sensor = world.spawn_actor(blueprint, spawn_point)

        # lidar data
        self.points = None
        self.obj_idx = None
        self.obj_tag = None

        self.timestamp = None
        self.frame = 0
        # open3d point cloud object
        self.o3d_pointcloud = o3d.geometry.PointCloud()

        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: SemanticLidarSensor._on_data_event(
                weak_self, event))

    @staticmethod
    def _on_data_event(weak_self, event):
        """Semantic Lidar  method"""
        self = weak_self()
        if not self:
            return

        # shape:(n, 6)
        data = np.frombuffer(event.raw_data, dtype=np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('CosAngle', np.float32), ('ObjIdx', np.uint32),
            ('ObjTag', np.uint32)]))

        # (x, y, z, intensity)
        self.points = np.array([data['x'], data['y'], data['z']]).T
        self.obj_tag = np.array(data['ObjTag'])
        self.obj_idx = np.array(data['ObjIdx'])

        self.data = data
        self.frame = event.frame
        self.timestamp = event.timestamp


class PerceptionManager:
    """
    Default perception module. Currenly only used to detect vehicles.

    Parameters
    ----------
    vehicle : carla.Vehicle
        carla Vehicle, we need this to spawn sensors.

    config_yaml : dict
        Configuration dictionary for perception.

    cav_world : opencda object
        CAV World object that saves all cav information, shared ML model,
         and sumo2carla id mapping dictionary.

    data_dump : bool
        Whether dumping data, if true, semantic lidar will be spawned.

    carla_world : carla.world
        CARLA world, used for rsu.

    Attributes
    ----------
    lidar : opencda object
        Lidar sensor manager.

    rgb_camera : opencda object
        RGB camera manager.

    o3d_vis : o3d object
        Open3d point cloud visualizer.
    """

    def __init__(self, vehicle, config_yaml, cav_world,
                 data_dump=False, carla_world=None, infra_id=None):
        self.vehicle = vehicle
        self.carla_world = carla_world if carla_world is not None \
            else self.vehicle.get_world()
        self.id = infra_id if infra_id is not None else vehicle.id

        self.activate = config_yaml['activate']
        self.camera_visualize = config_yaml['camera_visualize']
        self.camera_num = min(config_yaml['camera_num'], 4)
        self.lidar_visualize = config_yaml['lidar_visualize']
        self.global_position = config_yaml['global_position'] \
            if 'global_position' in config_yaml else None


        self.prev_timestamp = 0
        # self.cur_frame={} #set()
        self.pre_frame={} #set()
        self.carlaId_Num_dict = {}
        self.index=1
        data_dump = True
        # self.seg_cam_bp = blueprint_library.find('sensor.camera.semantic_segmentation')

        self.cav_world = weakref.ref(cav_world)()
        ml_manager = cav_world.ml_manager

        # if self.activate and data_dump:
        #     sys.exit("When you dump data, please deactivate the "
        #              "detection function for precise label.")

        if self.activate and not ml_manager:
            sys.exit(
                'If you activate the perception module, '
                'then apply_ml must be set to true in'
                'the argument parser to load the detection DL model.')
        self.ml_manager = ml_manager

        self._safety_distance_threshold = 0.05
        self._front_unlabeled_distance = 0.0

        self.gl_proj = None

        # we only spawn the camera when perception module is activated or
        # camera visualization is needed
        if self.activate or self.camera_visualize:
            self.rgb_camera = []
            mount_position = ['front', 'right', 'left', 'back']
            for i in range(self.camera_num):
                self.rgb_camera.append(
                    CameraSensor(
                        vehicle, self.carla_world, mount_position[i],
                        self.global_position))

        else:
            self.rgb_camera = None

        # we only spawn the LiDAR when perception module is activated or lidar
        # visualization is needed
        if self.activate or self.lidar_visualize:
            self.lidar = LidarSensor(vehicle,
                                     self.carla_world,
                                     config_yaml['lidar'],
                                     self.global_position)
            self.o3d_vis = o3d_visualizer_init(self.id)
            # self.o3d_vis_color = o3d_visualizer_init(self.id)
        else:
            self.lidar = None
            self.o3d_vis = None
            self.o3d_vis_color = None

        # if data dump is true, semantic lidar is also spawned
        self.data_dump = data_dump
        if True:  # data_dump
            self.semantic_lidar = SemanticLidarSensor(vehicle,
                                                      self.carla_world,
                                                      config_yaml['lidar'],
                                                      self.global_position)

            self.semantic_camera = SemanticCameraSensor(vehicle,self.carla_world,None,self.global_position)
            self.depth_camera = DepthCameraSensor(vehicle,self.carla_world,None,self.global_position)


        # count how many steps have been passed
        self.count = 0
        # ego position
        self.ego_pos = None

        # the dictionary contains all objects
        self.objects = {}

        self.tick_record = False
        self.next_time = 0
        self.prev_time =0
        self.start_time=0
        self.first_tick = True

    def dist(self, a):
        """
        A fast method to retrieve the obstacle distance the ego
        vehicle from the server directly.

        Parameters
        ----------
        a : carla.actor
            The obstacle vehicle.

        Returns
        -------
        distance : float
            The distance between ego and the target actor.
        """

        return a.get_location().distance(self.ego_pos.location)

    def detect(self, ego_pos):
        """
        Detect surrounding objects. Currently only vehicle detection supported.

        Parameters
        ----------
        ego_pos : carla.Transform
            Ego vehicle pose.

        Returns
        -------
        objects : list
            A list that contains all detected obstacle vehicles.

        """
        self.ego_pos = ego_pos

        objects = {'vehicles': [],
                   'static': [],
                   'traffic_lights': [],
                   'fence':[]}

        if not self.activate:
            objects = self.deactivate_mode(objects)

        else:
            objects = self.activate_mode(objects)

        self.count += 1

        return objects


    def recordCavSurrounding(self,objects):
        lines = ""
        rows = 0
        vehicles = objects['vehicles']
        peds = []
        if 'ped' in objects.keys():
            peds = objects['ped']
        _len = len(vehicles) + len(peds)
        if _len == 0:
            return lines,rows

        cur_timestamp = 0
        if CarlaDataProvider.get_world():
            snapshot = CarlaDataProvider.get_world().get_snapshot()
            if snapshot:
                cur_timestamp = snapshot.timestamp.frame/100.0
                if self.tick_record ==  False:
                    self.next_time = cur_timestamp + 0.05
                    self.prev_time = cur_timestamp
                    self.tick_record = True

        if cur_timestamp >= self.next_time:
            time = self.next_time - self.prev_time
            if self.first_tick:
                self.start_time = 0.0
                self.first_tick = False
            else:
                self.start_time = self.start_time + time

            tmp = {}
            for cur_vehicle in vehicles:
                # loc = cur_vehicle.location
                # vec = cur_vehicle.velocity
                # vec_ = get_speed(cur_vehicle)

                v_id = cur_vehicle.v_id

                if cur_vehicle.v_id in self.pre_frame.keys():
                    pre_vehicle=self.pre_frame[v_id]

                    cur_loc = cur_vehicle.location
                    cur_vec = cur_vehicle.velocity
                    cur_vec_ = get_speed(cur_vehicle)

                    pre_loc = pre_vehicle.location
                    pre_vec = pre_vehicle.velocity
                    pre_vec_ = get_speed(pre_vehicle)
                    yaw = azimuthangle(cur_loc.x,cur_loc.y,pre_loc.x,pre_loc.y)


                    # rot = vehicle.transform.rotation
                    carla_id = pre_vehicle.carla_id
                    agentNo = self.carlaId_Num_dict[carla_id]
                    agentType = CarlaDataProvider.get_world().get_actor(carla_id).type_id

                    agentTypeNo = blueprints[agentType]

                    line = "0,0,{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},0\n".format(agentNo,carla_id,agentType,agentTypeNo,round(pre_loc.x,3),round(pre_loc.y,3),round(pre_loc.z,3),round(yaw*3.1415926/180,4),round(pre_vec.x,3),round(pre_vec.y,3),round(pre_vec.z,3),round(pre_vec_,3),round(self.start_time ,3),round(cur_timestamp,3))
                    lines += line
                    rows += 1
                    self.tick_record = False
                tmp[v_id] = cur_vehicle
                #########

                # rot = vehicle.transform.rotation
                # carla_id = vehicle.carla_id
                # agentNo = self.carlaId_Num_dict[carla_id]
                # agentType = CarlaDataProvider.get_world().get_actor(carla_id).type_id
                #
                # agentTypeNo = blueprints[agentType]
                #
                # line = "0,0,{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},0\n".format(agentNo,carla_id,agentType,agentTypeNo,round(loc.x,3),round(loc.y,3),round(loc.z,3),round(yaw*3.1415926/180,3),round(vec.x,3),round(vec.y,3),round(vec.z,3),round(vec_,3),round(self.start_time ,3),round(cur_timestamp,3))
                # lines += line
                # rows += 1
                # self.tick_record = False


            self.pre_frame = tmp
        return lines, rows

    def activate_mode(self, objects):
        """
        Use Yolov5 + Lidar fusion to detect objects.

        Parameters
        ----------
        objects : dict
            The dictionary that contains all category of detected objects.
            The key is the object category name and value is its 3d coordinates
            and confidence.

        Returns
        -------
         objects: dict
            Updated object dictionary.
        """
        # retrieve current cameras and lidar data
        rgb_images = []
        for rgb_camera in self.rgb_camera:
            while rgb_camera.image is None:
                continue
            rgb_images.append(
                cv2.cvtColor(
                    np.array(
                        rgb_camera.image),
                    cv2.COLOR_BGR2RGB))

        # yolo detection
        yolo_detection = self.ml_manager.detect_objects(rgb_images)
        # rgb_images for drawing

        rgb_draw_images = []
        if yolo_detection is None or len(yolo_detection) <=0:
            return objects
        for (i, rgb_camera) in enumerate(self.rgb_camera):
            # lidar projection
            rgb_image, projected_lidar = st.project_lidar_to_camera(
                self.lidar.sensor,
                rgb_camera.sensor, self.lidar.data, np.array(
                    rgb_camera.image))
            rgb_draw_images.append(rgb_image)

            # camera lidar fusion
            objects = o3d_camera_lidar_fusion(
                objects,
                yolo_detection[i],
                self.lidar.data,
                projected_lidar,
                self.lidar.sensor)



            # self.cur_frame.clear()
            # if len(objects['vehicles'])>0:
            #     for obj in objects['vehicles']:
            #         self.cur_frame[obj.v_id] = obj #.add(yolo[5])

            # calculate the speed. current we retrieve from the server
            # directly.
            self.speed_retrieve(objects)



        if self.camera_visualize:
            names = ['front', 'right', 'left', 'back']
            for (i, rgb_image) in enumerate(rgb_draw_images):
                if i > self.camera_num - 1 or i > self.camera_visualize - 1:
                    break
                rgb_image = self.ml_manager.draw_2d_box(
                    yolo_detection, rgb_image, i)

                # if i == 2:
                #     cv2.imwrite('1.png', rgb_image)
                # rgb_image = cv2.resize(rgb_image, (0, 0), fx=0.4, fy=0.4)

                cv2.imshow(
                    '%s camera of actor %d, perception activated' %
                    (names[i], self.id), rgb_image)
            cv2.waitKey(1)

        if self.lidar_visualize:
            while self.lidar.data is None:
                continue
            # o3d_pointcloud_encode(self.lidar.data, self.lidar.o3d_pointcloud)
            # o3d_visualizer_show(
            #     self.o3d_vis,
            #     self.count,
            #     self.lidar.o3d_pointcloud,
            #     objects)



            rgb_image_front = self.rgb_camera[0].image
            _, projected_lidar = st.project_lidar_to_camera(
                self.lidar.sensor,
                self.rgb_camera[0].sensor, self.lidar.data, np.array(
                    rgb_image_front))
            color_pcd,points = o3d_pointcloud_encode_color(self.lidar.data, self.lidar.o3d_pointcloud, projected_lidar, rgb_image_front)
            o3d_visualizer_show(self.o3d_vis,#_color
                self.count,
                self.lidar.o3d_pointcloud, objects)




            # add traffic light
        objects = self.retrieve_traffic_lights(objects)

        objects = self.retrieve_barrier(objects)

        #for test


        self.objects = objects



        return objects

    @staticmethod
    def distance(this, other):
        dx = this.x - other.x
        dy = this.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def deactivate_mode(self, objects):
        """
        Object detection using server information directly.

        Parameters
        ----------
        objects : dict
            The dictionary that contains all category of detected objects.
            The key is the object category name and value is its 3d coordinates
            and confidence.

        Returns
        -------
         objects: dict
            Updated object dictionary.
        """
        world = self.carla_world

        vehicle_list = world.get_actors().filter("*vehicle*")
        thresh = 50 if not self.data_dump else 120

        vehicle_list = [v for v in vehicle_list if self.dist(v) < thresh] # and v.id != self.id


        # use semantic lidar to filter out vehicles out of the range
        # if self.data_dump:
        #     vehicle_list = self.filter_vehicle_out_sensor(vehicle_list)

        # convert carla.Vehicle to opencda.ObstacleVehicle if lidar
        # visualization is required.
        if self.lidar:
            vehicle_list = [
                ObstacleVehicle(
                    None,
                    None,
                    v,
                    self.lidar.sensor,
                    self.cav_world.sumo2carla_ids) for v in vehicle_list]
        else:
            vehicle_list_=[]
            for v in vehicle_list:
                vehicle_list_.append(ObstacleVehicle(
                    None,
                    None,
                    v,
                    None,
                    self.cav_world.sumo2carla_ids))
                if v.id not in self.carlaId_Num_dict:
                    self.carlaId_Num_dict[v.id] = self.index
                    self.index = self.index + 1
            vehicle_list = vehicle_list_
            # vehicle_list = [
            #     ObstacleVehicle(
            #         None,
            #         None,
            #         v,
            #         None,
            #         self.cav_world.sumo2carla_ids) for v in vehicle_list]

        objects.update({'vehicles': vehicle_list})

        if self.camera_visualize:
            while self.rgb_camera[0].image is None:
                continue

            names = ['front', 'right', 'left', 'back']

            for (i, rgb_camera) in enumerate(self.rgb_camera):
                if i > self.camera_num - 1 or i > self.camera_visualize - 1:
                    break
                # we only visualiz the frontal camera
                rgb_image = np.array(rgb_camera.image)
                # draw the ground truth bbx on the camera image
                rgb_image = self.visualize_3d_bbx_front_camera(objects,
                                                               rgb_image,
                                                               i)
                # resize to make it fittable to the screen
                rgb_image = cv2.resize(rgb_image, (0, 0), fx=0.4, fy=0.4)

                # show image using cv2
                cv2.imshow(
                    '%s camera of actor %d, perception deactivated' %
                    (names[i], self.id), rgb_image)
                cv2.waitKey(1)

        if self.lidar_visualize:
            while self.lidar.data is None:
                continue
            o3d_pointcloud_encode(self.lidar.data, self.lidar.o3d_pointcloud)
            # render the raw lidar
            o3d_visualizer_show(
                self.o3d_vis,
                self.count,
                self.lidar.o3d_pointcloud,
                objects)

        # add traffic light
        objects = self.retrieve_traffic_lights(objects)

        objects = self.retrieve_barrier(objects)

        self.objects = objects

        return objects

    def get_front_unlabeled_distance(self, observation):
        width = observation.shape[1]
        xMin = 3 * width // 8
        xMax = 5 * width // 8

        return np.min(observation[:, xMin:xMax])


    # Get indices of unlabeled.
    def get_unlabeled_indices(self,segmentation_image):
        return np.where(np.all(segmentation_image == UNLABELED_SEGMENTATION_COLOR, axis=-1))

    # Get normalized depth values of vehicle pixels.
    def get_normalized_depth_of_unlabeled(self,depth_image, unlabeled_indices):
        height = depth_image.shape[0]
        width = depth_image.shape[1]
        observation = np.full(shape=(height, width, 1), fill_value=1, dtype=np.float32)

        for y, x in zip(unlabeled_indices[0], unlabeled_indices[1]):
            observation[y][x][0] = depth_image[y][x][0] / 255

        return observation

    def filter_fence_out_sensor_by_semantic_depth(self):
        unlabeled_indices = self.get_unlabeled_indices(self.semantic_camera.data)
        observation = self.get_normalized_depth_of_unlabeled(self.depth_camera.data, unlabeled_indices)

        fence_unique_id = list(np.unique(observation))

        fence_list = []

        for fence in fence_unique_id:
            fence_list.append(fence)

        # self._front_unlabeled_distance = self.get_front_unlabeled_distance(observation)

        # fence_list = []
        # print(self._front_unlabeled_distance)
        # if self._front_unlabeled_distance > self._safety_distance_threshold:
        #     pass
        # else:
        #     fence_obstacle = FenceObstacle(self._front_unlabeled_distance)
        #     fence_list.append(fence_obstacle)


        # print(distance)
        observation_image = observation * 255
        # cv2.imshow('observation_image',observation_image)
        # cv2.imwrite('1.png', observation_image)
        # cv2.waitKey(2)
        # if self.semantic_camera.data:
        #     cv2.imshow('segmentation', self.semantic_camera.data)
        #     cv2.waitKey(2)

        return fence_list

    def filter_fence_out_sensor(self):
        semantic_idx = self.semantic_lidar.obj_idx
        semantic_tag = self.semantic_lidar.obj_tag

        new_fence_list = []

        if semantic_idx is None or semantic_tag is None:
            return new_fence_list

        #not actor, 0
        # label 10 is the vehicle
        fence_idx = semantic_idx[semantic_tag == 0]  #semantic_tag == 2 or
        # each individual instance id

        # kinds=list(np.unique(semantic_idx))
        fence_unique_id = list(np.unique(fence_idx))

        print(fence_unique_id)
        # new_vehicle_list = []
        for fence in fence_unique_id:

            new_fence_list.append(fence)

        return new_fence_list

    def filter_vehicle_out_sensor(self, vehicle_list):
        """
        By utilizing semantic lidar, we can retrieve the objects that
        are in the lidar detection range from the server.
        This function is important for collect training data for object
        detection as it can filter out the objects out of the senor range.

        Parameters
        ----------
        vehicle_list : list
            The list contains all vehicles information retrieves from the
            server.

        Returns
        -------
        new_vehicle_list : list
            The list that filters out the out of scope vehicles.

        """
        semantic_idx = self.semantic_lidar.obj_idx
        semantic_tag = self.semantic_lidar.obj_tag

        new_vehicle_list = []

        if semantic_idx is None or semantic_tag is None:
            return new_vehicle_list

        # label 10 is the vehicle
        vehicle_idx = semantic_idx[semantic_tag == 10]
        # each individual instance id
        vehicle_unique_id = list(np.unique(vehicle_idx))

        # new_vehicle_list = []
        for veh in vehicle_list:
            if veh.id in vehicle_unique_id:
                new_vehicle_list.append(veh)

        # print(new_vehicle_list)
        return new_vehicle_list

    def visualize_3d_bbx_front_camera(self, objects, rgb_image, camera_index):
        """
        Visualize the 3d bounding box on frontal camera image.

        Parameters
        ----------
        objects : dict
            The object dictionary.

        rgb_image : np.ndarray
            Received rgb image at current timestamp.

        camera_index : int
            Indicate the index of the current camera.

        """
        camera_transform = \
            self.rgb_camera[camera_index].sensor.get_transform()
        camera_location = \
            camera_transform.location
        camera_rotation = \
            camera_transform.rotation

        for v in objects['vehicles']:
            # we only draw the bounding box in the fov of camera
            _, angle = cal_distance_angle(
                v.get_location(), camera_location,
                camera_rotation.yaw)
            if angle < 60:
                bbx_camera = st.get_2d_bb(
                    v,
                    self.rgb_camera[camera_index].sensor,
                    camera_transform)
                cv2.rectangle(rgb_image,
                              (int(bbx_camera[0, 0]), int(bbx_camera[0, 1])),
                              (int(bbx_camera[1, 0]), int(bbx_camera[1, 1])),
                              (255, 0, 0), 2)

        return rgb_image

    def rotate_retrieve(self, objects):
        if 'vehicles' not in objects:
            return

        world = self.carla_world
        vehicle_list = world.get_actors().filter("*vehicle*")
        vehicle_list = [v for v in vehicle_list if self.dist(v) < 50 and
                        v.id != self.id]

        for v in vehicle_list:
            loc = v.get_location()
            for obstacle_vehicle in objects['vehicles']:
                obstacle_loc = obstacle_vehicle.get_location()
                if abs(loc.x - obstacle_loc.x) <= 3.0 and \
                        abs(loc.y - obstacle_loc.y) <= 3.0:
                    obstacle_vehicle.set_transform(v.get_transform())

    def speed_retrieve_v(self, objects):
        pass

    def speed_retrieve(self, objects):
        """
        We don't implement any obstacle speed calculation algorithm.
        The speed will be retrieved from the server directly.

        Parameters
        ----------
        objects : dict
            The dictionary contains the objects.
        """
        if 'vehicles' not in objects:
            return

        world = self.carla_world
        vehicle_list = world.get_actors().filter("*vehicle*")
        vehicle_list = [v for v in vehicle_list if self.dist(v) < 80 and
                        v.id != self.id]

        # todo: consider the minimum distance to be safer in next version

        for v in vehicle_list:
            loc = v.get_location()
            for obstacle_vehicle in objects['vehicles']:
                obstacle_speed = get_speed(obstacle_vehicle)
                # if speed > 0, it represents that the vehicle
                # has been already matched.
                if obstacle_speed > 0:
                    continue
                obstacle_loc = obstacle_vehicle.get_location()
                if abs(loc.x - obstacle_loc.x) <= 3 and \
                        abs(loc.y - obstacle_loc.y) <= 3:
                    obstacle_vehicle.set_velocity(v.get_velocity())
                    obstacle_vehicle.set_transform(v.get_transform())
                    # the case where the obstacle vehicle is controled by
                    # sumo
                    if self.cav_world.sumo2carla_ids:
                        sumo_speed = \
                            get_speed_sumo(self.cav_world.sumo2carla_ids,
                                           v.id)
                        if sumo_speed > 0:
                            # todo: consider the yaw angle in the future
                            speed_vector = carla.Vector3D(sumo_speed, 0, 0)
                            obstacle_vehicle.set_velocity(speed_vector)
                    if v.id not in self.carlaId_Num_dict:
                        self.carlaId_Num_dict[v.id]=self.index
                        self.index = self.index+1
                    obstacle_vehicle.set_carla_id(v.id)

    def retrieve_barrier(self, objects):

        world = self.carla_world

        # fence_list = self.filter_fence_out_sensor_by_semantic_depth()

        # if len(fence_list) == 0:
        fence_list = self.filter_fence_out_sensor()

        # convert carla.Vehicle to opencda.ObstacleVehicle if lidar
        # visualization is required.
        # if self.lidar:
        #     fence_list = [
        #         StaticObstacle(
        #             None,
        #             None,
        #             v,
        #             self.lidar.sensor,
        #             self.cav_world.sumo2carla_ids) for v in fence_list]
        # else:
        #     fence_list = [
        #         StaticObstacle(
        #             None,
        #             None,
        #             v,
        #             None,
        #             self.cav_world.sumo2carla_ids) for v in fence_list]

        objects.update({'fence': fence_list})

        return objects

    def retrieve_traffic_lights(self, objects):
        """
        Retrieve the traffic lights nearby from the server  directly.
        Next version may consider add traffic light detection module.

        Parameters
        ----------
        objects : dict
            The dictionary that contains all objects.

        Returns
        -------
        object : dict
            The updated dictionary.
        """
        world = self.carla_world
        tl_list = world.get_actors().filter('traffic.traffic_light*')

        objects.update({'traffic_lights': []})

        for tl in tl_list:
            distance = self.dist(tl)
            if distance < 50:
                traffic_light = TrafficLight(tl.get_location(),
                                             tl.get_state())
                objects['traffic_lights'].append(traffic_light)
        return objects

    def destroy(self):
        """
        Destroy sensors.
        """
        self.index = 1
        self.carlaId_Num_dict = {}

        if self.rgb_camera:
            for rgb_camera in self.rgb_camera:
                rgb_camera.sensor.destroy()

        if self.lidar:
            self.lidar.sensor.destroy()

        if self.camera_visualize:
            cv2.destroyAllWindows()

        if self.lidar_visualize:
            self.o3d_vis.destroy_window()

        if self.data_dump:
            self.semantic_lidar.sensor.destroy()
