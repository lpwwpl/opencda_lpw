#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Module used to parse all the route and scenario configuration parameters.
"""

import json
import math
import xml.etree.ElementTree as ET

import carla
from agents.navigation.local_planner import RoadOption
from opencda.scenarioconfigs.route_scenario_configuration import RouteScenarioConfiguration

# TODO  check this threshold, it could be a bit larger but not so large that we cluster scenarios.
TRIGGER_THRESHOLD = 2.0  # Threshold to say if a trigger position is new or repeated, works for matching positions
TRIGGER_ANGLE_THRESHOLD = 10  # Threshold to say if two angles can be considering matching when matching transforms.


class RouteParser(object):

    """
    Pure static class used to parse all the route and scenario configuration parameters.
    """

    @staticmethod
    def parse_annotations_file(annotation_filename):
        """
        Return the annotations of which positions where the scenarios are going to happen.
        :param annotation_filename: the filename for the anotations file
        :return:
        """

        with open(annotation_filename, 'r') as f:
            annotation_dict = json.loads(f.read())

        final_dict = {}

        for town_dict in annotation_dict['available_scenarios']:
            final_dict.update(town_dict)

        return final_dict  # the file has a current maps name that is an one element vec




    @staticmethod
    def check_trigger_position(new_trigger, existing_triggers):
        """
        Check if this trigger position already exists or if it is a new one.
        :param new_trigger:
        :param existing_triggers:
        :return:
        """

        for trigger_id in existing_triggers.keys():
            trigger = existing_triggers[trigger_id]
            dx = trigger[0] - new_trigger[0]
            dy = trigger[1] - new_trigger[1]
            distance = math.sqrt(dx * dx + dy * dy)

            dyaw = (trigger[3] - new_trigger[3]) % 360
            if distance < TRIGGER_THRESHOLD \
                    and (dyaw < TRIGGER_ANGLE_THRESHOLD or dyaw > (360 - TRIGGER_ANGLE_THRESHOLD)):
                return trigger_id

        return None

    @staticmethod
    def convert_waypoint_float(waypoint):
        """
        Convert waypoint values to float
        """
        waypoint['x'] = float(waypoint['x'])
        waypoint['y'] = float(waypoint['y'])
        waypoint['z'] = float(waypoint['z'])
        waypoint['yaw'] = float(waypoint['yaw'])

    @staticmethod
    def match_world_location_to_route(world_location, route_description):
        """
        We match this location to a given route.
            world_location:
            route_description:
        """
        def match_waypoints(waypoint1, wtransform):
            """
            Check if waypoint1 and wtransform are similar
            """
            dx = float(waypoint1['x']) - wtransform.location.x
            dy = float(waypoint1['y']) - wtransform.location.y
            dz = float(waypoint1['z']) - wtransform.location.z
            dpos = math.sqrt(dx * dx + dy * dy + dz * dz)

            dyaw = (float(waypoint1['yaw']) - wtransform.rotation.yaw) % 360

            return dpos < TRIGGER_THRESHOLD \
                and (dyaw < TRIGGER_ANGLE_THRESHOLD or dyaw > (360 - TRIGGER_ANGLE_THRESHOLD))

        match_position = 0
        # TODO this function can be optimized to run on Log(N) time
        for route_waypoint in route_description:
            if match_waypoints(world_location, route_waypoint[0]):
                return match_position
            match_position += 1

        return None

    @staticmethod
    def get_scenario_type(scenario, match_position, trajectory):
        """
        Some scenarios have different types depending on the route.
        :param scenario: the scenario name
        :param match_position: the matching position for the scenarion
        :param trajectory: the route trajectory the ego is following
        :return: tag representing this subtype

        Also used to check which are not viable (Such as an scenario
        that triggers when turning but the route doesnt')
        WARNING: These tags are used at:
            - VehicleTurningRoute
            - SignalJunctionCrossingRoute
        and changes to these tags will affect them
        """

        def check_this_waypoint(tuple_wp_turn):
            """
            Decides whether or not the waypoint will define the scenario behavior
            """
            if RoadOption.LANEFOLLOW == tuple_wp_turn[1]:
                return False
            elif RoadOption.CHANGELANELEFT == tuple_wp_turn[1]:
                return False
            elif RoadOption.CHANGELANERIGHT == tuple_wp_turn[1]:
                return False
            return True

        # Unused tag for the rest of scenarios,
        # can't be None as they are still valid scenarios
        subtype = 'valid'

        if scenario == 'Scenario4':
            for tuple_wp_turn in trajectory[match_position:]:
                if check_this_waypoint(tuple_wp_turn):
                    if RoadOption.LEFT == tuple_wp_turn[1]:
                        subtype = 'S4left'
                    elif RoadOption.RIGHT == tuple_wp_turn[1]:
                        subtype = 'S4right'
                    else:
                        subtype = None
                    break  # Avoid checking all of them
                subtype = None

        if scenario == 'Scenario7':
            for tuple_wp_turn in trajectory[match_position:]:
                if check_this_waypoint(tuple_wp_turn):
                    if RoadOption.LEFT == tuple_wp_turn[1]:
                        subtype = 'S7left'
                    elif RoadOption.RIGHT == tuple_wp_turn[1]:
                        subtype = 'S7right'
                    elif RoadOption.STRAIGHT == tuple_wp_turn[1]:
                        subtype = 'S7opposite'
                    else:
                        subtype = None
                    break  # Avoid checking all of them
                subtype = None

        if scenario == 'Scenario8':
            for tuple_wp_turn in trajectory[match_position:]:
                if check_this_waypoint(tuple_wp_turn):
                    if RoadOption.LEFT == tuple_wp_turn[1]:
                        subtype = 'S8left'
                    else:
                        subtype = None
                    break  # Avoid checking all of them
                subtype = None

        if scenario == 'Scenario9':
            for tuple_wp_turn in trajectory[match_position:]:
                if check_this_waypoint(tuple_wp_turn):
                    if RoadOption.RIGHT == tuple_wp_turn[1]:
                        subtype = 'S9right'
                    else:
                        subtype = None
                    break  # Avoid checking all of them
                subtype = None

        return subtype

    @staticmethod
    def scan_route_for_scenarios(annotations):
        """
        Just returns a plain list of possible scenarios that can happen in this route by matching
        the locations from the scenario into the route description

        :return:  A list of scenario definitions with their correspondent parameters
        """

        # the triggers dictionaries:
        existent_triggers = {}
        # We have a table of IDs and trigger positions associated
        possible_scenarios = {}

        # Keep track of the trigger ids being added
        latest_trigger_id = 0

        scenarios = annotations.scenario_config
        for scenario in scenarios:  # For each existent scenario
            if scenario['scenario_type'] is None:
                break
            scenario_name = scenario["scenario_type"]
            for i, event in enumerate(scenario['trigger']):
                waypoint = event['transform']  # trigger point of this scenario
                # RouteParser.convert_waypoint_float(waypoint)
                # We match trigger point to the  route, now we need to check if the route affects
                match_position = waypoint
                # RouteParser.match_world_location_to_route(
                #    waypoint, trajectory)
                if match_position is not None:
                    # We match a location for this scenario, create a scenario object so this scenario
                    # can be instantiated later

                    other_vehicles = scenario['other_actors']
                    scenario_subtype = 'valid'
                    # RouteParser.get_scenario_type(scenario_name, match_position,
                    #                                                 trajectory)
                    if scenario_subtype is None:
                        continue
                    scenario_description = {
                        'name': scenario_name,
                        'other_actors': other_vehicles,
                        'trigger': waypoint,
                        'scenario_type': scenario_subtype,  # some scenarios have route dependent configs
                    }

                    trigger_id = RouteParser.check_trigger_position(waypoint, existent_triggers)
                    if trigger_id is None:
                        # This trigger does not exist create a new reference on existent triggers
                        existent_triggers.update({latest_trigger_id: waypoint})
                        # Update a reference for this trigger on the possible scenarios
                        possible_scenarios.update({latest_trigger_id: []})
                        trigger_id = latest_trigger_id
                        # Increment the latest trigger
                        latest_trigger_id += 1

                    possible_scenarios[trigger_id].append(scenario_description)


        return possible_scenarios, existent_triggers
