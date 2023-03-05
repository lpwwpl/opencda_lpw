# -*- coding: utf-8 -*-
"""
Basic class of CAV
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import uuid
import opencda.scenario_testing.utils.customized_map_api as map_api
from opencda.core.actuation.control_manager \
    import ControlManager
from opencda.core.application.platooning.platoon_behavior_agent\
    import PlatooningBehaviorAgent
from opencda.core.common.v2x_manager \
    import V2XManager
from opencda.core.sensing.localization.localization_manager \
    import LocalizationManager
from opencda.core.sensing.perception.perception_manager \
    import PerceptionManager
from opencda.core.plan.behavior_agent \
    import BehaviorAgent
from opencda.core.map.map_manager import MapManager
from opencda.core.common.data_dumper import DataDumper
# from opencda.test.scenarios_manager import ScenariosManager
from opencda.tools.route_parser import RouteParser
from opencda.tools.route_manipulation import interpolate_trajectory
from opencda.scenariomanager.carla_data_provider import CarlaDataProvider
from opencda.scenarios.route_scenario import RouteScenario
from opencda.scenariomanager.scenario_manager import ScenarioManager
import opencda.scenario_testing.utils.sim_api as sim_api
import carla
from opencda.scenarioconfigs.route_scenario_configuration import RouteScenarioConfiguration

from opencda.core.sensing.perception.obstacle_vehicle import \
    ObstacleVehicle

class VehicleManager(object):
    """
    A class manager to embed different modules with vehicle together.

    Parameters
    ----------
    vehicle : carla.Vehicle
        The carla.Vehicle. We need this class to spawn our gnss and imu sensor.

    config_yaml : dict
        The configuration dictionary of this CAV.

    application : list
        The application category, currently support:['single','platoon'].

    carla_map : carla.Map
        The CARLA simulation map.

    cav_world : opencda object
        CAV World. This is used for V2X communication simulation.

    current_time : str
        Timestamp of the simulation beginning, used for data dumping.

    data_dumping : bool
        Indicates whether to dump sensor data during simulation.

    Attributes
    ----------
    v2x_manager : opencda object
        The current V2X manager.

    localizer : opencda object
        The current localization manager.

    perception_manager : opencda object
        The current V2X perception manager.

    agent : opencda object
        The current carla agent that handles the basic behavior
         planning of ego vehicle.

    controller : opencda object
        The current control manager.

    data_dumper : opencda object
        Used for dumping sensor data.
    """

    def __init__(
            self,
            # vehicle,
            config_yaml,
            application,
            carla_map,
            town,
            cav_world,
            current_time='',vehicle=None,
            data_dumping=True):

        # an unique uuid for this vehicle
        self.vid = str(uuid.uuid1())
        # self.vehicle = vehicle
        self.carla_map = carla_map


        # retrieve the configure for different modules
        sensing_config = config_yaml['sensing']
        map_config = config_yaml['map_manager']
        behavior_config = config_yaml['behavior']
        control_config = config_yaml['controller']
        v2x_config = config_yaml['v2x']

        self.manager = None
        ########################
        self.vehicle = None
        trajectory = []
        if 'scenarios' in config_yaml.keys():
            scenarios_config = config_yaml['scenarios']
            self.manager = ScenarioManager()

            # if 'spawn_special' not in config_yaml:
                # spawn_transform = carla.Transform(
                #     carla.Location(
                #         x=config_yaml['spawn_position'][0],
                #         y=config_yaml['spawn_position'][1],
                #         z=config_yaml['spawn_position'][2]),
                #     carla.Rotation(
                #         pitch=config_yaml['spawn_position'][5],
                #         yaw=config_yaml['spawn_position'][4],
                #         roll=config_yaml['spawn_position'][3]))
            #
                # spawn_transform = carla.Location(
                #         x=config_yaml['spawn_position'][0],
                #         y=config_yaml['spawn_position'][1],
                #         z=config_yaml['spawn_position'][2])

            # destination = carla.Transform(carla.Location(x=config_yaml['destination'][0],
            #                              y=config_yaml['destination'][1],
            #                              z=config_yaml['destination'][2]),
            #         carla.Rotation(
            #             pitch=config_yaml['destination'][5],
            #             yaw=config_yaml['destination'][4],
            #             roll=config_yaml['destination'][3]))
            #
            # destination = carla.Location(x=config_yaml['destination'][0],
            #                              y=config_yaml['destination'][1],
            #                              z=config_yaml['destination'][2])

            route_config = config_yaml['route']
            for waypoint in route_config:
                Loc = carla.Location(
                            x=waypoint['waypoint'][0],
                            y=waypoint['waypoint'][1],
                            z=waypoint['waypoint'][2])
                trajectory.append(Loc)

            new_config = RouteScenarioConfiguration()
            new_config.scenario_config = scenarios_config
            new_config.town = town
            new_config.hop = behavior_config['sample_resolution']
            new_config.name = "RouteScenario_{}".format(town)
            # new_config.weather = RouteParser.parse_weather(route)

            # waypoint_list = []  # the list of waypoints that can be found on this route
            # for waypoint in route.iter('waypoint'):
            #     waypoint_list.append(carla.Location(x=float(waypoint.attrib['x']),
            #                                         y=float(waypoint.attrib['y']),
            #                                         z=float(waypoint.attrib['z'])))
            # new_config.trajectory = waypoint_list

            # self.ego_start = spawn_transform
            # self.ego_dest = destination

            # 3.append(self.ego_start)
            # trajectory.append(self.ego_dest)


            new_config.trajectory = trajectory

            # CarlaDataProvider.
            self.scenario = RouteScenario(world=CarlaDataProvider.get_world(), config=new_config, debug_mode=True)
            self.manager.load_scenario(self.scenario)
            vehicle = self.scenario.ego_vehicles[0]
        elif vehicle is None:
            default_model = 'vehicle.tesla.model3'

            cav_vehicle_bp = \
                CarlaDataProvider.get_world().get_blueprint_library().find(default_model)
            cav_vehicle_bp.set_attribute('color', '0, 0, 255')
            spawn_transform=None
            if 'spawn_special' not in config_yaml:
                spawn_transform = carla.Transform(
                    carla.Location(
                        x=config_yaml['spawn_position'][0],
                        y=config_yaml['spawn_position'][1],
                        z=config_yaml['spawn_position'][2]),
                    carla.Rotation(
                        pitch=config_yaml['spawn_position'][5],
                        yaw=config_yaml['spawn_position'][4],
                        roll=config_yaml['spawn_position'][3]))
            else:
                carla_version = '0.9.10'
                spawn_transform = map_api.spawn_helper_2lanefree(carla_version,
                                             *config_yaml['spawn_special'])

            vehicle = CarlaDataProvider.request_new_actor(default_model,spawn_transform,rolename='ego_vehicle')
            #CarlaDataProvider.get_world().spawn_actor(cav_vehicle_bp, spawn_transform)

        else:
            pass
        self.vehicle = vehicle
        # v2x module
        self.v2x_manager = V2XManager(cav_world, v2x_config, self.vid)
        # localization module
        self.localizer = LocalizationManager(
            vehicle, sensing_config['localization'], carla_map)
        # perception module
        self.perception_manager = PerceptionManager(
            vehicle, sensing_config['perception'], cav_world,
            data_dumping)
        # map manager
        self.map_manager = MapManager(vehicle,
                                      carla_map,
                                      map_config)

        # behavior agent
        self.agent = None
        if 'platooning' in application:
            platoon_config = config_yaml['platoon']
            self.agent = PlatooningBehaviorAgent(
                vehicle,
                self,
                self.v2x_manager,
                behavior_config,
                platoon_config,
                carla_map)
        else:
            self.agent = BehaviorAgent(vehicle, carla_map, behavior_config)

        # Control module
        self.controller = ControlManager(control_config)

        if data_dumping:
            self.data_dumper = DataDumper(self.perception_manager,
                                          vehicle.id,
                                          save_time=current_time)
        else:
            self.data_dumper = None

        cav_world.update_vehicle_manager(self)

        self.objects = None


    def set_destination(
            self,
            start_location,
            end_location,
            clean=False,
            end_reset=True):
        """
        Set global route.

        Parameters
        ----------
        start_location : carla.location
            The CAV start location.

        end_location : carla.location
            The CAV destination.

        clean : bool
             Indicator of whether clean waypoint queue.

        end_reset : bool
            Indicator of whether reset the end location.

        Returns
        -------
        """

        self.agent.set_destination(
            start_location, end_location, clean, end_reset)

    def record_surrounding(self):
        record,rows = self.perception_manager.recordCavSurrounding(self.objects)
        return record,rows

    def update_info(self):
        """
        Call perception and localization module to
        retrieve surrounding info an ego position.
        """
        # localization
        self.localizer.localize()

        ego_pos = self.localizer.get_ego_pos()
        ego_spd = self.localizer.get_ego_spd()

        # print(ego_pos)
        # object detection
        objects = self.perception_manager.detect(ego_pos)

        ###########


        self.objects = objects
        obstacle_vehicle = ObstacleVehicle(None, None, self.vehicle)
        obstacle_vehicle.set_v_id(-1)
        if 'vehicles' in self.objects:
            if self.vehicle.id not in self.perception_manager.carlaId_Num_dict:
                self.perception_manager.carlaId_Num_dict[self.vehicle.id] = self.perception_manager.index
                self.perception_manager.index = self.perception_manager.index + 1
            self.objects['vehicles'].append(obstacle_vehicle)

        ############################

        # update the ego pose for map manager
        self.map_manager.update_information(ego_pos)

        # update ego position and speed to v2x manager,
        # and then v2x manager will search the nearby cavs
        self.v2x_manager.update_info(ego_pos, ego_spd)


        self.agent.update_information(ego_pos, ego_spd, objects)
        # pass position and speed info to controller
        self.controller.update_info(ego_pos, ego_spd)


        # for actor in actors:
        #     if actor.attributes.get('type') == 'ego_vehicle':
        #         continue
        #     distance = ego_vehicle.get_location().distance(carla.Location(actor))
        #     if distance<3.0:
        #         trans = actor.get_transform()
        #         # line = CarlaDataProvider.get_location(actor)
        #         vel = actor.get_velocity()
        #         vel_ = CarlaDataProvider.get_velocity(actor)
        #         model = ""
        #         testNo=0
        #         repeatNo=0
        #         agentNo=0
        #         agentID=0
        #         agentType=0
        #         agentTypeNo=0
        #         sim_time=0
        #         time=0
        #         fps =0
        #         lines.append(line)

    def run_step(self, target_speed=None):
        """
        Execute one step of navigation.
        """
        # visualize the bev map if needed
        self.map_manager.run_step()
        target_speed, target_pos = self.agent.run_step(target_speed)
        control = self.controller.run_step(target_speed, target_pos)

        if self.manager is not None:
    ######################################
            timestamp = None
            self.manager._running = True
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                self.manager._tick_scenario(timestamp)


    ######################################

        # dump data
        if self.data_dumper:
            self.data_dumper.run_step(self.perception_manager,
                                      self.localizer,
                                      self.agent)

        return control

    def destroy(self):
        """
        Destroy the actor vehicle
        """
        self.perception_manager.destroy()
        self.localizer.destroy()

        self.vehicle.destroy()
        self.map_manager.destroy()
