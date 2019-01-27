#!/usr/bin/python3
import random

SEPARATOR = "_"
DISCOUNT_RATE = 0.7


class RLBase:
    """
    Base class for rl_model_based (Model Based RL) and rl_model_free (Model Free
    RL). This class provides common methods and variables shared by the two 
    classes
    """

    def __init__(self, simulator):
        """
        Constructor of a model based reinforcement learning
        """
        # Environment
        self._simulator = simulator
        self._track = simulator.track
        self._car = simulator.car
        # Time
        self._t = 0
        # Discount factor
        self._gamma = 0.7

        # actions_set is all the possible actions the agent could take in the
        # environment
        self._actions_set = set()
        self.init_actions_set()

        # states_set is all the possible state in the given environment and
        # agent. ie. track and car.
        self._states_set = set()
        self.init_states_set()

    
    def get_attributes(self):
        """
        Get the attribute of model
        """
        return {
            "Discount Factor": self._gamma
        }


    def init_states_set(self):
        """
        Get all the possible states in the problem. For this problem, a state
        for any given point on the track is a variation of the velocity.
        """
        track_map = self._track.get_map()
        velocity_ranges = self._car.get_velocity_range()

        for row_index in range(len(track_map)):
            for col_index in range(len(track_map[row_index])):
                # If the coordinate is not a wall on the map, then it should
                # have a state.
                if not self._track.isWall(col_index, row_index):
                    for velocity_x in velocity_ranges[0]:
                        for velocity_y in velocity_ranges[1]:
                            from_state = self.encode_state(
                                col_index, row_index, velocity_x, velocity_y)
                            self.add_state(from_state)


    def add_state(self, from_state):
        """
        Add the given state to the state set for each valid action
        """
        # skip the states that any action won't be able avoid a crash
        for action in self._actions_set:
            if not self.will_crash(from_state, action):
                self._states_set.add(from_state)
                break


    def encode_state(self, x, y, vx, vy):
        """
        Get the encoded state by concating coordinate x, y and velocity x, y 
        using underscore. ie. (x, y, vx, vy) => x_y_vx_vy
        """
        return SEPARATOR.join(str(item) for item in [x, y, vx, vy])

    def decode_state(self, encoded_state):
        """
        Decode the given encoded state into parameters
        ie. (x, y, vx, vy) <= x_y_vx_vy
        """
        splitted = encoded_state.split(SEPARATOR)
        return {
            "x": int(splitted[0]),
            "y": int(splitted[1]),
            "vx": int(splitted[2]),
            "vy": int(splitted[3]),
        }

    def init_actions_set(self):
        """
        Initialize the actions set with all the possible action combination for
        x and y direction. 
        """
        acceleration_options = self._car.get_acceleration_options()
        for ax in acceleration_options[0]:
            for ay in acceleration_options[1]:
                self._actions_set.add(self.encode_action(ax, ay))

    def encode_action(self, ax, ay):
        """
        Get the encoded action by concating action ax and ay. 
        ie. (ax, ay) => ax_ay
        """
        return SEPARATOR.join([str(ax), str(ay)])

    def decode_action(self, encoded_action):
        """
        Decode the given encoded action into parameters
        ie. (ax, ay) <= ax_ay
        """
        splitted = encoded_action.split(SEPARATOR)
        return {
            "ax": int(splitted[0]),
            "ay": int(splitted[1])
        }

    def simulate(self, from_state, action):
        """
        Simulate the action is applied for a car at the from_state
        """
        from_decoded = self.decode_state(from_state)
        action_decoded = self.decode_action(action)
        self._simulator.simulate(
            from_decoded.get("x"), from_decoded.get("y"),
            from_decoded.get("vx"), from_decoded.get("vy"),
            action_decoded.get("ax"), action_decoded.get("ay"))

    def reward(self, from_state, action):
        """
        Calculate the reward for the from state and action
        """
        if self.will_finish(from_state, action):
            # Use 0 to incentivize navigation to finish line
            return 0
        return -1

    def will_crash(self, from_state, action):
        """
        Simulate to see if car will crash
        """
        self.simulate(from_state, action)
        return self._simulator.has_crashed()


    def will_finish(self, from_state, action):
        """
        Simulate to see if car will finish
        """
        self.simulate(from_state, action)
        return self._simulator.has_finished()

    def will_stay(self, from_state, action):
        """
        """
        self.simulate(from_state, action)
        to_state = self.get_cur_state()
        return from_state == to_state

    def get_cur_state(self):
        """
        Get the curernt state of the car
        """
        x, y = self._simulator.car.get_position()
        vx, vy = self._simulator.car.get_velocity()
        from_state = self.encode_state(x, y, vx, vy)
        return from_state