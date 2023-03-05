# -*- coding: utf-8 -*-

# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import importlib

class CavWorld(object):
    """
    A customized world object to save all CDA vehicle
    information and shared ML models. During co-simulation,
    it is also used to save the sumo-carla id mapping.

    Parameters
    ----------
    apply_ml : bool
        Whether apply ml/dl models in this simulation, please make sure
        you have install torch/sklearn before setting this to True.

    Attributes
    ----------
    vehicle_id_set : set
        A set that stores vehicle IDs.

    _vehicle_manager_dict : dict
        A dictionary that stores vehicle managers.

    _platooning_dict : dict
        A dictionary that stores platooning managers.

    _rsu_manager_dict : dict
        A dictionary that stores RSU managers.

    ml_manager : opencda object.
        The machine learning manager class.
    """

    def __init__(self, apply_ml=False):

        self.vehicle_id_set = set()
        self._vehicle_manager_dict = {}
        self._platooning_dict = {}
        self._rsu_manager_dict = {}
        self.ml_manager = None

        if apply_ml:
            # we import in this way so the user don't need to install ml
            # packages unless they require to
            ml_manager = getattr(importlib.import_module(
                "opencda.customize.ml_libs.ml_manager"), 'MLManager')
            # initialize the ml manager to load the DL/ML models into memory
            self.ml_manager = ml_manager()

        # this is used only when co-simulation activated.
        self.sumo2carla_ids = {}

    def update_vehicle_manager(self, vehicle_manager):
        """
        Update created CAV manager to the world.

        Parameters
        ----------
        vehicle_manager : opencda object
            The vehicle manager class.
        """
        self.vehicle_id_set.add(vehicle_manager.vehicle.id)
        self._vehicle_manager_dict.update(
            {vehicle_manager.vid: vehicle_manager})

    def update_platooning(self, platooning_manger):
        """
        Add created platooning.

        Parameters
        ----------
        platooning_manger : opencda object
            The platooning manager class.
        """
        self._platooning_dict.update(
            {platooning_manger.pmid: platooning_manger})

    def update_rsu_manager(self, rsu_manager):
        """
        Add rsu manager.

        Parameters
        ----------
        rsu_manager : opencda object
            The RSU manager class.
        """
        self._rsu_manager_dict.update({rsu_manager.rid: rsu_manager})

    def update_sumo_vehicles(self, sumo2carla_ids):
        """
        Update the sumo carla mapping dict. This is only called
        when cosimulation is conducted.

        Parameters
        ----------
        sumo2carla_ids : dict
            Key is sumo id and value is carla id.
        """
        self.sumo2carla_ids = sumo2carla_ids

    def get_vehicle_managers(self):
        """
        Return vehicle manager dictionary.
        """
        return self._vehicle_manager_dict

    def get_platoon_dict(self):
        """
        Return existing platoons.
        """
        return self._platooning_dict

    def locate_vehicle_manager(self, loc):
        """
        Locate the vehicle manager based on the given location.

        Parameters
        ----------
        loc : carla.Location
            Vehicle location.

        Returns
        -------
        target_vm : opencda object
            The vehicle manager at the give location.
        """

        target_vm = None
        for key, vm in self._vehicle_manager_dict.items():
            x = vm.localizer.get_ego_pos().location.x
            y = vm.localizer.get_ego_pos().location.y

            if loc.x == x and loc.y == y:
                target_vm = vm
                break

        return target_vm

    # @staticmethod
    # def get_actor_by_id(actor_id):
    #     """
    #     Get an actor from the pool by using its ID. If the actor
    #     does not exist, None is returned.
    #     """
    #     if actor_id in CavWorld._carla_actor_pool:
    #         return CavWorld._carla_actor_pool[actor_id]
    #
    #     print("Non-existing actor id {}".format(actor_id))
    #     return None
    #
    # @staticmethod
    # def remove_actor_by_id(actor_id):
    #     """
    #     Remove an actor from the pool using its ID
    #     """
    #     if actor_id in CavWorld._carla_actor_pool:
    #         CavWorld._carla_actor_pool[actor_id].destroy()
    #         CavWorld._carla_actor_pool[actor_id] = None
    #         CavWorld._carla_actor_pool.pop(actor_id)
    #     else:
    #         print("Trying to remove a non-existing actor id {}".format(actor_id))
    #
    # @staticmethod
    # def request_new_actor(model, spawn_point, rolename='scenario', autopilot=False,
    #                       random_location=False, color=None, actor_category="car"):
    #     """
    #     This method tries to create a new actor, returning it if successful (None otherwise).
    #     """
    #     blueprint = CavWorld.create_blueprint(model, rolename, color, actor_category)
    #
    #     if random_location:
    #         actor = None
    #         while not actor:
    #             spawn_point = CavWorld._rng.choice(CavWorld._spawn_points)
    #             actor = CavWorld._world.try_spawn_actor(blueprint, spawn_point)
    #
    #     else:
    #         # slightly lift the actor to avoid collisions with ground when spawning the actor
    #         # DO NOT USE spawn_point directly, as this will modify spawn_point permanently
    #         _spawn_point = carla.Transform(carla.Location(), spawn_point.rotation)
    #         _spawn_point.location.x = spawn_point.location.x
    #         _spawn_point.location.y = spawn_point.location.y
    #         _spawn_point.location.z = spawn_point.location.z + 0.2
    #         actor = CavWorld._world.try_spawn_actor(blueprint, _spawn_point)
    #
    #     if actor is None:
    #         raise RuntimeError(
    #             "Error: Unable to spawn vehicle {} at {}".format(blueprint.id, spawn_point))
    #     else:
    #         # Let's deactivate the autopilot of the actor if it belongs to vehicle
    #         if actor in CavWorld._blueprint_library.filter('vehicle.*'):
    #             actor.set_autopilot(autopilot)
    #         else:
    #             pass
    #
    #     # wait for the actor to be spawned properly before we do anything
    #     # if CarlaDataProvider.is_sync_mode():
    #     #     CarlaDataProvider._world.tick()
    #     # else:
    #     #     CarlaDataProvider._world.wait_for_tick()
    #
    #     if actor is None:
    #         return None
    #
    #     CavWorld._carla_actor_pool[actor.id] = actor
    #     CavWorld.register_actor(actor)
    #     return actor
    #
    # @staticmethod
    # def cleanup():
    #     """
    #     Cleanup and remove all entries from all dictionaries
    #     """
    #     DestroyActor = carla.command.DestroyActor       # pylint: disable=invalid-name
    #     batch = []
    #
    #     for actor_id in CavWorld._carla_actor_pool.copy():
    #         actor = CavWorld._carla_actor_pool[actor_id]
    #         if actor.is_alive:
    #             batch.append(DestroyActor(actor))
    #
    #     CavWorld._world = None
    #     CavWorld._rng = random.RandomState(CavWorld._random_seed)
    #
    # @staticmethod
    # def actor_id_exists(actor_id):
    #     """
    #     Check if a certain id is still at the simulation
    #     """
    #     if actor_id in CavWorld._carla_actor_pool:
    #         return True
    #
    #     return False
    #
    # @staticmethod
    # def get_actors():
    #     """
    #     Return list of actors and their ids
    #
    #     Note: iteritems from six is used to allow compatibility with Python 2 and 3
    #     """
    #     return iteritems(CavWorld._carla_actor_pool)
    #
    # @staticmethod
    # def create_blueprint(model, rolename='scenario', color=None, actor_category="car"):
    #     """
    #     Function to setup the blueprint of an actor given its model and other relevant parameters
    #     """
    #
    #     _actor_blueprint_categories = {
    #         'car': 'vehicle.tesla.model3',
    #         'van': 'vehicle.volkswagen.t2',
    #         'truck': 'vehicle.carlamotors.carlacola',
    #         'trailer': '',
    #         'semitrailer': '',
    #         'bus': 'vehicle.volkswagen.t2',
    #         'motorbike': 'vehicle.kawasaki.ninja',
    #         'bicycle': 'vehicle.diamondback.century',
    #         'train': '',
    #         'tram': '',
    #         'pedestrian': 'walker.pedestrian.0001',
    #     }
    #
    #     # Set the model
    #     try:
    #         blueprint = CavWorld._rng.choice(CavWorld._blueprint_library.filter(model))
    #     except ValueError:
    #         # The model is not part of the blueprint library. Let's take a default one for the given category
    #         bp_filter = "vehicle.*"
    #         new_model = _actor_blueprint_categories[actor_category]
    #         if new_model != '':
    #             bp_filter = new_model
    #         print("WARNING: Actor model {} not available. Using instead {}".format(model, new_model))
    #         blueprint = CavWorld._rng.choice(CavWorld._blueprint_library.filter(bp_filter))
    #
    #     # Set the color
    #     if color:
    #         if not blueprint.has_attribute('color'):
    #             print(
    #                 "WARNING: Cannot set Color ({}) for actor {} due to missing blueprint attribute".format(
    #                     color, blueprint.id))
    #         else:
    #             default_color_rgba = blueprint.get_attribute('color').as_color()
    #             default_color = '({}, {}, {})'.format(default_color_rgba.r, default_color_rgba.g, default_color_rgba.b)
    #             try:
    #                 blueprint.set_attribute('color', color)
    #             except ValueError:
    #                 # Color can't be set for this vehicle
    #                 print("WARNING: Color ({}) cannot be set for actor {}. Using instead: ({})".format(
    #                     color, blueprint.id, default_color))
    #                 blueprint.set_attribute('color', default_color)
    #     else:
    #         if blueprint.has_attribute('color') and rolename != 'hero':
    #             color = CavWorld._rng.choice(blueprint.get_attribute('color').recommended_values)
    #             blueprint.set_attribute('color', color)
    #
    #     # Make pedestrians mortal
    #     if blueprint.has_attribute('is_invincible'):
    #         blueprint.set_attribute('is_invincible', 'false')
    #
    #     # Set the rolename
    #     if blueprint.has_attribute('role_name'):
    #         blueprint.set_attribute('role_name', rolename)
    #
    #     return blueprint
    #
    # @staticmethod
    # def set_world(world):
    #     """
    #     Set the world and world settings
    #     """
    #     CavWorld._world = world
    #     CavWorld._blueprint_library = world.get_blueprint_library()
    #     CavWorld.generate_spawn_points()
    #
    # @staticmethod
    # def get_world():
    #     """
    #     Return world
    #     """
    #     return CavWorld._world