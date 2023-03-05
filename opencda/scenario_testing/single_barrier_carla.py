# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import carla
import numpy as np
import opencda.scenario_testing.utils.sim_api as sim_api
from opencda.core.common.cav_world import CavWorld
from opencda.scenario_testing.evaluations.evaluate_manager import \
    EvaluationManager
from opencda.scenario_testing.utils.yaml_utils import load_yaml
from PyQt5 import QtWidgets, uic, QtGui, QtCore
import sys
import pyqtgraph.opengl as gl
class Qt3D_Diag(QtWidgets.QDialog):
    def __init__(self):
        self.diag = QtWidgets.QDialog()
        self.layout = QtWidgets.QHBoxLayout
        self.diag.setLayout(self.layout)
        self.gl_proj = None

    def updateData(self,points,color_pcd):
        if self.gl_proj is None:
            self.gl_proj = gl.GLScatterPlotItem(
                pos=points,
                size=1 * np.ones((points.shape[0])) * 0.01,
                # size=ps_size,
                color=color_pcd,
                pxMode=False)
            self.gl_proj.setGLOptions('opaque')
            self.layout.addItem(self.gl_proj)
            self.diag.show()
        else:
            self.gl_proj.setData(
                pos=points,
                size=1 * np.ones((points.shape[0])) * 0.01,
                # size=ps_size, # - 10 ms
                color=color_pcd
            )



def run_scenario(opt, config_yaml):
    try:
        scenario_params = load_yaml(config_yaml)

        # create CAV world
        cav_world = CavWorld(opt.apply_ml)

        # create scenario manager
        scenario_manager = sim_api.ScenarioManager(scenario_params,
                                                   opt.apply_ml,
                                                   opt.version,
                                                   town='barrier',
                                                   cav_world=cav_world)

        if opt.record:
            scenario_manager.client. \
                start_recorder("single_barrier_carla.log", True)

        single_cav_list = \
            scenario_manager.create_vehicle_manager(application=['single'])

        # create background traffic in carla
        traffic_manager, bg_veh_list = \
            scenario_manager.create_traffic_carla()

        # create evaluation manager
        eval_manager = \
            EvaluationManager(scenario_manager.cav_world,
                              script_name='single_2lanefree_carla',
                              current_time=scenario_params['current_time'])

        spectator = scenario_manager.world.get_spectator()

        # diag = Qt3D_Diag()

        # diag.show()
        # run steps
        while True:
            scenario_manager.tick()
            transform = single_cav_list[0].vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                transform.location +
                carla.Location(
                    z=50),
                carla.Rotation(
                    pitch=-
                    90)))

            for i, single_cav in enumerate(single_cav_list):
                single_cav.update_info()
                control = single_cav.run_step()
                single_cav.vehicle.apply_control(control)

    finally:
        eval_manager.evaluate()

        if opt.record:
            scenario_manager.client.stop_recorder()

        scenario_manager.close()

        for v in single_cav_list:
            v.destroy()
        for v in bg_veh_list:
            v.destroy()

