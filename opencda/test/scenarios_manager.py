
from opencda.scenariomanager.carla_data_provider import CarlaDataProvider

class ScenariosManager(object):

    def __init__(self, cav_world, config_yaml):
        # - trigger_transform: [ -15.63,-124.89,1.0,-180 ]
        #   scenario_type: Scenario3
        #   other_actors: ~
        trigger = config_yaml['trigger_transform']
        type = config_yaml['scenario_type']
        others = config_yaml['other_actors']

    def _initialize_actors(self, config):
        """
        Default initialization of other actors.
        Override this method in child class to provide custom initialization.
        """
        if config.other_actors:
            new_actors = CarlaDataProvider.request_new_actors(config.other_actors)
            if not new_actors:
                raise Exception("Error: Unable to add actors")

            for new_actor in new_actors:
                self.other_actors.append(new_actor)

    def remove_all_actors(self):
        """
        Remove all actors
        """
        for i, _ in enumerate(self.other_actors):
            if self.other_actors[i] is not None:
                if CarlaDataProvider.actor_id_exists(self.other_actors[i].id):
                    CarlaDataProvider.remove_actor_by_id(self.other_actors[i].id)
                self.other_actors[i] = None
        self.other_actors = []

    def update_info(self, ego_pos, ego_spd):
        pass