#!/usr/bin/env python

# Copyright (c) 2019-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides Challenge routes as standalone scenarios
"""

from __future__ import print_function

import math
import traceback
import xml.etree.ElementTree as ET
import numpy.random as random

import py_trees

import carla

from agents.navigation.local_planner import RoadOption

# pylint: disable=line-too-long
from opencda.scenarioconfigs.scenario_configuration import ScenarioConfiguration, ActorConfigurationData
# pylint: enable=line-too-long
from opencda.scenariomanager.carla_data_provider import CarlaDataProvider
from opencda.scenariomanager.scenarioatomics.atomic_behaviors import Idle, ScenarioTriggerer
from opencda.scenarios.basic_scenario import BasicScenario

from opencda.scenarios.control_loss import ControlLoss
from opencda.scenarios.follow_leading_vehicle import FollowLeadingVehicle
from opencda.scenarios.object_crash_vehicle import DynamicObjectCrossing
from opencda.scenarios.object_crash_intersection import VehicleTurningRoute
from opencda.scenarios.other_leading_vehicle import OtherLeadingVehicle
from opencda.scenarios.maneuver_opposite_direction import ManeuverOppositeDirection
from opencda.scenarios.junction_crossing_route import SignalJunctionCrossingRoute, NoSignalJunctionCrossingRoute
from opencda.tools.route_parser import RouteParser,TRIGGER_THRESHOLD,TRIGGER_ANGLE_THRESHOLD
from opencda.tools.route_manipulation import interpolate_trajectory

SECONDS_GIVEN_PER_METERS = 0.4

NUMBER_CLASS_TRANSLATION = {
    "Scenario1": ControlLoss,
    "Scenario2": FollowLeadingVehicle,
    "Scenario3": DynamicObjectCrossing,
    "Scenario4": VehicleTurningRoute,
    "Scenario5": OtherLeadingVehicle,
    "Scenario6": ManeuverOppositeDirection,
    "Scenario7": SignalJunctionCrossingRoute,
    "Scenario8": SignalJunctionCrossingRoute,
    "Scenario9": SignalJunctionCrossingRoute,
    "Scenario10": NoSignalJunctionCrossingRoute
}


def convert_json_to_transform(actor_dict):
    """
    Convert a JSON string to a CARLA transform
    """
    return carla.Transform(location=carla.Location(x=float(actor_dict[0]), y=float(actor_dict[1]),
                                                   z=float(actor_dict[2])),
                           rotation=carla.Rotation(roll=0.0, pitch=0.0, yaw=float(actor_dict[3])))


def convert_json_to_actor(actor_dict):
    """
    Convert a JSON string to an ActorConfigurationData dictionary
    """
    node = ET.Element('waypoint')
    node.set('x', actor_dict[0])
    node.set('y', actor_dict[1])
    node.set('z', actor_dict[2])
    node.set('yaw', actor_dict[3])

    return ActorConfigurationData.parse_from_node(node, 'simulation')


def convert_transform_to_location(transform_vec):
    """
    Convert a vector of transforms to a vector of locations
    """
    location_vec = []
    for transform_tuple in transform_vec:
        location_vec.append((transform_tuple[0].location, transform_tuple[1]))

    return location_vec


def compare_scenarios(scenario_choice, existent_scenario):
    """
    Compare function for scenarios based on distance of the scenario start position
    """
    def transform_to_pos_vec(scenario):
        """
        Convert left/right/front to a meaningful CARLA position
        """
        position_vec = [scenario['trigger']]
        if scenario['other_actors'] is not None:
            if 'left' in scenario['other_actors']:
                position_vec += [scenario['other_actors']['left']]
            if 'front' in scenario['other_actors']:
                position_vec += [scenario['other_actors']['front']]
            if 'right' in scenario['other_actors']:
                position_vec += [scenario['other_actors']['right']]

        return position_vec

    # put the positions of the scenario choice into a vec of positions to be able to compare

    choice_vec = transform_to_pos_vec(scenario_choice)
    existent_vec = transform_to_pos_vec(existent_scenario)
    for pos_choice in choice_vec:
        for pos_existent in existent_vec:

            dx = float(pos_choice[0]) - float(pos_existent[0])
            dy = float(pos_choice[1]) - float(pos_existent[0])
            dz = float(pos_choice[2]) - float(pos_existent[0])
            dist_position = math.sqrt(dx * dx + dy * dy + dz * dz)
            dyaw = float(pos_choice[3]) - float(pos_choice[3])
            dist_angle = math.sqrt(dyaw * dyaw)
            if dist_position < TRIGGER_THRESHOLD and dist_angle < TRIGGER_ANGLE_THRESHOLD:
                return True

    return False


class RouteScenario(BasicScenario):

    """
    Implementation of a RouteScenario, i.e. a scenario that consists of driving along a pre-defined route,
    along which several smaller scenarios are triggered
    """

    def __init__(self, world, config, debug_mode=False, timeout=300):
        """
        Setup all relevant parameters and create scenarios along route
        """

        self.config = config
        self.route = None
        self.timeout = timeout
        self.sampled_scenarios_definitions = None

        self._update_route(world, config, debug_mode)

        ego_vehicle = self._update_ego_vehicle()
        # self.ego_vehicle = ego_vehicle

        # CarlaDataProvider._carla_actor_pool[ego_vehicle.id] = ego_vehicle
        # CarlaDataProvider.register_actor(ego_vehicle)
        self.list_scenarios = self._build_scenario_instances(world,
                                                             ego_vehicle,
                                                             self.sampled_scenarios_definitions,
                                                             scenarios_per_tick=5,
                                                             timeout=self.timeout,
                                                             debug_mode=debug_mode)

        super(RouteScenario, self).__init__(name=config.name,
                                            ego_vehicles=[ego_vehicle],
                                            config=config,
                                            world=world,
                                            debug_mode=True,
                                            terminate_on_failure=False)

    # def create_scn_manager(self,cav_world,scenarios_config):
    #     scns_list = []
    #     # default_model = 'vehicle.lincoln.mkz_2017'
    #
    #     # cav_vehicle_bp = \
    #     #     self.world.get_blueprint_library().find(default_model)
    #
    #     potential_scenarios_definitions, _ = RouteParser.scan_route_for_scenarios(scenarios_config)

    def _update_route(self, world, config, debug_mode):
        """
        Update the input route, i.e. refine waypoint list, and extract possible scenario locations

        Parameters:
        - world: CARLA world
        - config: Scenario configuration (RouteConfiguration)
        """
        potential_scenarios_definitions, _ = RouteParser.scan_route_for_scenarios(config)
        gps_route, route = interpolate_trajectory(world, config.trajectory,config.hop)
        self.route = route
        CarlaDataProvider.set_ego_vehicle_route(convert_transform_to_location(self.route))
        self.sampled_scenarios_definitions = self._scenario_sampling(potential_scenarios_definitions)
        # Transform the scenario file into a dictionary
        # world_annotations = RouteParser.parse_annotations_file(config.scenario_file)
        #
        # # prepare route's trajectory (interpolate and add the GPS route)
        # gps_route, route = None,None #interpolate_trajectory(world, config.trajectory)
        #
        # potential_scenarios_definitions, _ = RouteParser.scan_route_for_scenarios(config.town, route, world_annotations)
        #
        # self.route = route
        # CarlaDataProvider.set_ego_vehicle_route(convert_transform_to_location(self.route))
        #
        # config.agent.set_global_plan(gps_route, self.route)
        #
        # # Sample the scenarios to be used for this route instance.
        # self.sampled_scenarios_definitions = self._scenario_sampling(potential_scenarios_definitions)
        #
        # # Timeout of scenario in seconds
        # self.timeout = self._estimate_route_timeout()
        #
        # Print route in debug mode
        if debug_mode:
            self._draw_waypoints(world, self.route, vertical_shift=1.0, persistency=50000.0)

    def _update_ego_vehicle(self):
        """
        Set/Update the start position of the ego_vehicle
        """
        # move ego to correct position
        elevate_transform = self.route[0][0]
        elevate_transform.location.z += 0.5

        ego_vehicle = CarlaDataProvider.request_new_actor('vehicle.lincoln.mkz2017',
                                                          elevate_transform,
                                                          rolename='hero')

        return ego_vehicle

    def _estimate_route_timeout(self):
        """
        Estimate the duration of the route
        """
        route_length = 0.0  # in meters

        prev_point = self.route[0][0]
        for current_point, _ in self.route[1:]:
            dist = current_point.location.distance(prev_point.location)
            route_length += dist
            prev_point = current_point

        return int(SECONDS_GIVEN_PER_METERS * route_length)

    # pylint: disable=no-self-use
    def _draw_waypoints(self, world, waypoints, vertical_shift, persistency=-1):
        """
        Draw a list of waypoints at a certain height given in vertical_shift.
        """
        for w in waypoints:
            wp = w[0].location + carla.Location(z=vertical_shift)

            size = 0.2
            if w[1] == RoadOption.LEFT:  # Yellow
                color = carla.Color(255, 255, 0)
            elif w[1] == RoadOption.RIGHT:  # Cyan
                color = carla.Color(0, 255, 255)
            elif w[1] == RoadOption.CHANGELANELEFT:  # Orange
                color = carla.Color(255, 64, 0)
            elif w[1] == RoadOption.CHANGELANERIGHT:  # Dark Cyan
                color = carla.Color(0, 64, 255)
            elif w[1] == RoadOption.STRAIGHT:  # Gray
                color = carla.Color(128, 128, 128)
            else:  # LANEFOLLOW
                color = carla.Color(0, 255, 0)  # Green
                size = 0.1

            world.debug.draw_point(wp, size=size, color=color, life_time=persistency)

        world.debug.draw_point(waypoints[0][0].location + carla.Location(z=vertical_shift), size=0.2,
                               color=carla.Color(0, 0, 255), life_time=persistency)
        world.debug.draw_point(waypoints[-1][0].location + carla.Location(z=vertical_shift), size=0.2,
                               color=carla.Color(255, 0, 0), life_time=persistency)

    def _scenario_sampling(self, potential_scenarios_definitions, random_seed=0):
        """
        The function used to sample the scenarios that are going to happen for this route.
        """

        # fix the random seed for reproducibility
        rng = random.RandomState(random_seed)

        def position_sampled(scenario_choice, sampled_scenarios):
            """
            Check if a position was already sampled, i.e. used for another scenario
            """
            for existent_scenario in sampled_scenarios:
                # If the scenarios have equal positions then it is true.
                if compare_scenarios(scenario_choice, existent_scenario):
                    return True

            return False

        # The idea is to randomly sample a scenario per trigger position.
        sampled_scenarios = []
        for trigger in potential_scenarios_definitions.keys():
            possible_scenarios = potential_scenarios_definitions[trigger]

            scenario_choice = rng.choice(possible_scenarios)
            del possible_scenarios[possible_scenarios.index(scenario_choice)]
            # We keep sampling and testing if this position is present on any of the scenarios.
            while position_sampled(scenario_choice, sampled_scenarios):
                if possible_scenarios is None or not possible_scenarios:
                    scenario_choice = None
                    break
                scenario_choice = rng.choice(possible_scenarios)
                del possible_scenarios[possible_scenarios.index(scenario_choice)]

            if scenario_choice is not None:
                sampled_scenarios.append(scenario_choice)

        return sampled_scenarios

    def _build_scenario_instances(self, world, ego_vehicle, scenario_definitions,
                                  scenarios_per_tick=5, timeout=300, debug_mode=False):
        """
        Based on the parsed route and possible scenarios, build all the scenario classes.
        """
        scenario_instance_vec = []


        if debug_mode:
            scenario_number=0
            for scenario in scenario_definitions:
                loc = carla.Location(scenario['trigger'][0],
                                     scenario['trigger'][1],
                                     scenario['trigger'][2]) + carla.Location(z=2.0)
                world.debug.draw_point(loc, size=0.3, color=carla.Color(255, 0, 0), life_time=100000)
                world.debug.draw_string(loc, str(scenario['name']), draw_shadow=False,
                                        color=carla.Color(0, 0, 255), life_time=100000, persistent_lines=True)


                # Get the class possibilities for this scenario number
                scenario_class = NUMBER_CLASS_TRANSLATION[scenario['name']]

                # Create the other actors that are going to appear
                if scenario['other_actors'] is not None:
                    list_of_actor_conf_instances = self._get_actors_instances(scenario['other_actors'])
                else:
                    list_of_actor_conf_instances = []
                # Create an actor configuration for the ego-vehicle trigger position

                egoactor_trigger_position = convert_json_to_transform(scenario['trigger'])
                scenario_configuration = ScenarioConfiguration()
                scenario_configuration.other_actors = list_of_actor_conf_instances
                scenario_configuration.trigger_points = [egoactor_trigger_position]
                scenario_configuration.subtype = scenario['scenario_type']
                scenario_configuration.ego_vehicles = [ActorConfigurationData('vehicle.lincoln.mkz2017',
                                                                              ego_vehicle.get_transform(),
                                                                              'hero')]
                route_var_name = "ScenarioRouteNumber{}".format(scenario_number)
                scenario_number=scenario_number+1
                scenario_configuration.route_var_name = route_var_name

                try:
                    scenario_instance = scenario_class(world, [ego_vehicle], scenario_configuration,
                                                       timeout=timeout)
                    # Do a tick every once in a while to avoid spawning everything at the same time
                    if scenario_number % scenarios_per_tick == 0:
                        if CarlaDataProvider.is_sync_mode():
                            world.tick()
                        else:
                            world.wait_for_tick()

                    scenario_number += 1
                except Exception as e:      # pylint: disable=broad-except
                    if debug_mode:
                        traceback.print_exc()
                    print("Skipping scenario '{}' due to setup error: {}".format(scenario['name'], e))
                    continue

                scenario_instance_vec.append(scenario_instance)

        return scenario_instance_vec

    def _get_actors_instances(self, list_of_antagonist_actors):
        """
        Get the full list of actor instances.
        """

        def get_actors_from_list(list_of_actor_def):
            """
                Receives a list of actor definitions and creates an actual list of ActorConfigurationObjects
            """
            sublist_of_actors = []
            #for actor_def in list_of_actor_def:
            sublist_of_actors.append(convert_json_to_actor(list_of_actor_def))

            return sublist_of_actors

        list_of_actors = []
        # Parse vehicles to the left
        if 'front' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['front'])

        if 'left' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['left'])

        if 'right' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['right'])

        return list_of_actors

    # pylint: enable=no-self-use

    def _initialize_actors(self, config):
        """
        Set other_actors to the superset of all scenario actors
        """

        # Create the background activity of the route
        town_amount = {
            'Town01': 120,
            'Town02': 100,
            'Town03': 120,
            'Town04': 200,
            'Town05': 120,
            'Town06': 150,
            'Town07': 110,
            'Town08': 180,
            'Town09': 300,
            'Town10': 120,
            'simple_signals': 0,
        }

        amount = town_amount[config.town] if config.town in town_amount else 0

        new_actors = CarlaDataProvider.request_new_batch_actors('vehicle.*',
                                                                amount,
                                                                carla.Transform(),
                                                                autopilot=True,
                                                                random_location=True,
                                                                rolename='background')

        if new_actors is None:
            raise Exception("Error: Unable to add the background activity, all spawn points were occupied")

        for _actor in new_actors:
            self.other_actors.append(_actor)

        # Add all the actors of the specific scenarios to self.other_actors
        for scenario in self.list_scenarios:
            self.other_actors.extend(scenario.other_actors)

    def _create_behavior(self):
        """
        Basic behavior do nothing, i.e. Idle
        """
        scenario_trigger_distance = 1.5  # Max trigger distance between route and scenario

        behavior = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        subbehavior = py_trees.composites.Parallel(name="Behavior",
                                                   policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)

        scenario_behaviors = []
        blackboard_list = []

        for i, scenario in enumerate(self.list_scenarios):
            if scenario.scenario.behavior is not None:
                route_var_name = scenario.config.route_var_name
                if route_var_name is not None:
                    scenario_behaviors.append(scenario.scenario.behavior)
                    blackboard_list.append([scenario.config.route_var_name,
                                            scenario.config.trigger_points[0].location])
                else:
                    name = "{} - {}".format(i, scenario.scenario.behavior.name)
                    oneshot_idiom = None #oneshot_behavior(name,
                                          #           behaviour=scenario.scenario.behavior,
                                           #          name=name)
                    scenario_behaviors.append(oneshot_idiom)

        # Add behavior that manages the scenarios trigger conditions
        scenario_triggerer = ScenarioTriggerer(
            self.ego_vehicles[0],
            self.route,
            blackboard_list,
            scenario_trigger_distance,
            repeat_scenarios=False
        )

        subbehavior.add_child(scenario_triggerer)  # make ScenarioTriggerer the first thing to be checked
        subbehavior.add_children(scenario_behaviors)
        subbehavior.add_child(Idle())  # The behaviours cannot make the route scenario stop
        behavior.add_child(subbehavior)

        return behavior



    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
