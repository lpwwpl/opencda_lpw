#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Basic CARLA Autonomous Driving training scenario
"""

import py_trees

from opencda.scenarioconfigs.route_scenario_configuration import RouteConfiguration
from opencda.scenariomanager.scenarioatomics.atomic_behaviors import Idle
from opencda.scenariomanager.scenarioatomics.atomic_criteria import (CollisionTest,
                                                                     InRouteTest,
                                                                     RouteCompletionTest,
                                                                     OutsideRouteLanesTest,
                                                                     RunningRedLightTest,
                                                                     RunningStopTest,
                                                                     ActorSpeedAboveThresholdTest)
from opencda.scenarios.basic_scenario import BasicScenario


class MasterScenario(BasicScenario):

    """
    Implementation of a  Master scenario that controls the route.

    This is a single ego vehicle scenario
    """

    radius = 10.0           # meters

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False,
                 timeout=300):
        """
        Setup all relevant parameters and create scenario
        """
        self.config = config
        self.route = None
        # Timeout of scenario in seconds
        self.timeout = timeout

        if hasattr(self.config, 'route'):
            self.route = self.config.route
        else:
            raise ValueError("Master scenario must have a route")

        super(MasterScenario, self).__init__("MasterScenario", ego_vehicles=ego_vehicles, config=config,
                                             world=world, debug_mode=debug_mode,
                                             terminate_on_failure=True)

    def _create_behavior(self):
        """
        Basic behavior do nothing, i.e. Idle
        """

        # Build behavior tree
        sequence = py_trees.composites.Sequence("MasterScenario")
        idle_behavior = Idle()
        sequence.add_child(idle_behavior)

        return sequence


    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
