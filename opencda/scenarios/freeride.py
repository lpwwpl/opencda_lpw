#!/usr/bin/env python

# Copyright (c) 2019-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Simple freeride scenario. No action, no triggers. Ego vehicle can simply cruise around.
"""

import py_trees

from opencda.scenariomanager.scenarioatomics.atomic_behaviors import Idle
from opencda.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from opencda.scenarios.basic_scenario import BasicScenario


class FreeRide(BasicScenario):

    """
    Implementation of a simple free ride scenario that consits only of the ego vehicle
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False,
                 timeout=10000000):
        """
        Setup all relevant parameters and create scenario
        """
        # Timeout of scenario in seconds
        self.timeout = timeout
        super(FreeRide, self).__init__("FreeRide",
                                       ego_vehicles,
                                       config,
                                       world,
                                       debug_mode)

    def _setup_scenario_trigger(self, config):
        """
        """
        return None

    def _create_behavior(self):
        """
        """
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        sequence.add_child(Idle())
        return sequence



    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
