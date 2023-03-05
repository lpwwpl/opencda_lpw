#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the ScenarioManager implementation.
It must not be modified and is for reference only!
"""

from __future__ import print_function
import sys
import time

import py_trees

# from opencda.autoagents.agent_wrapper import AgentWrapper
from opencda.scenariomanager.carla_data_provider import CarlaDataProvider
from opencda.scenariomanager.result_writer import ResultOutputProvider
from opencda.scenariomanager.timer import GameTime
from opencda.scenariomanager.watchdog import Watchdog


class ScenarioManager(object):

    """
    Basic scenario manager class. This class holds all functionality
    required to start, and analyze a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manager.run_scenario()
       This function is designed to explicitly control start and end of
       the scenario execution
    4. Trigger a result evaluation with manager.analyze_scenario()
    5. If needed, cleanup with manager.stop_scenario()
    """

    def __init__(self, debug_mode=True, sync_mode=False, timeout=2.0):
        """
        Setups up the parameters, which will be filled at load_scenario()

        """
        self.scenario = None
        self.scenario_tree = None
        self.scenario_class = None
        self.ego_vehicles = None
        self.other_actors = None

        self._debug_mode = debug_mode
        self._agent = None
        self._sync_mode = sync_mode
        self._running = False
        self._timestamp_last_run = 0.0
        self._timeout = timeout

        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None

    def _reset(self):
        """
        Reset all parameters
        """
        self._running = False
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        GameTime.restart()

    def cleanup(self):
        """
        This function triggers a proper termination of a scenario
        """

        if self.scenario is not None:
            self.scenario.terminate()

        if self._agent is not None:
            self._agent.cleanup()
            self._agent = None

        CarlaDataProvider.cleanup()

    def load_scenario(self, scenario):
        """
        Load a new scenario
        """
        self._reset()
        # self._agent = AgentWrapper(agent) if agent else None
        # if self._agent is not None:
        #     self._sync_mode = True
        self.scenario_class = scenario
        self.scenario = scenario.scenario
        self.scenario_tree = self.scenario.scenario_tree
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors

        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)

    def run_scenario(self):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        print("ScenarioManager: Running scenario {}".format(self.scenario_tree.name))
        self.start_system_time = time.time()
        start_game_time = GameTime.get_time()

        self._running = True

        while self._running:
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                self._tick_scenario(timestamp)


        self.cleanup()

        self.end_system_time = time.time()
        end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - \
            self.start_system_time
        self.scenario_duration_game = end_game_time - start_game_time

        if self.scenario_tree.status == py_trees.common.Status.FAILURE:
            print("ScenarioManager: Terminated due to failure")

    def _tick_scenario(self, timestamp):
        """
        Run next tick of scenario and the agent.
        If running synchornously, it also handles the ticking of the world.
        """

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()

            # if self._agent is not None:
            #     ego_action = self._agent()

            # if self._agent is not None:
            #     self.ego_vehicles[0].apply_control(ego_action)

            # Tick scenario
            self.scenario_tree.tick_once()

            if True: #self._debug_mode:
                print("\n")
                py_trees.display.print_ascii_tree(self.scenario_tree, show_status=True)
                sys.stdout.flush()

            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._running = False

        # if self._sync_mode and self._running and self._watchdog.get_status():
        #     CarlaDataProvider.get_world().tick()

    # def get_running_status(self):
    #     """
    #     returns:
    #        bool:  False if watchdog exception occured, True otherwise
    #     """
    #     return self._watchdog.get_status()

    def stop_scenario(self):
        """
        This function is used by the overall signal handler to terminate the scenario execution
        """
        self._running = False

