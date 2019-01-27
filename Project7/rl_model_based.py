#!/usr/bin/python3
import random
import rl_base

SEPARATOR = "_"
DISCOUNT_RATE = 0.7


class ReinformacementLearning(rl_base.RLBase): 

    def __init__(self, simulator):
        """
        Constructor of a model based reinforcement learning
        """
        rl_base.RLBase.__init__(self, simulator)
        # values_map is the value-functions from one state at time t to a real 
        # number
        self._values_map = {}
        self.init_values_map()
        # transition_map is the transition-function from one state to another
        # state with an action to a probability
        self._transition_map = {}
        self.init_transition_map()
        # quality_map is the q-function from one state and an action at time t 
        # to a real number
        self._quality_map = {}
        # policy_map is the pi-function from one state to an action
        self._policy_map = {}


    def value_iteration(self, epsilon):
        """
        Perform value iteration
        """
        max_value_diff = float("inf")
        while max_value_diff >= epsilon:
            self._t += 1
            for from_state in self._states_set:
                for encoded_action in self._actions_set:
                    # set quality for each state-action combination
                    self.set_quality(self._t, from_state, encoded_action)
                # update the policy and value using the max quality action at 
                # the current state
                max_qtsa_action, max_qtsa = self.get_max_quality(self._t, from_state)
                self.set_policy(from_state, max_qtsa_action)
                self.set_value(self._t, from_state, max_qtsa)
            max_value_diff = self.get_max_value_diff(self._t)
            # print(self._t)
        return self._policy_map


    def get_action(self):
        """
        Get an action to execute from the curren state
        """
        from_state = self.get_cur_state()
        action = self._policy_map.get(from_state)
        if action is None:
            return None, None, float("-inf")
        action_decoded = self.decode_action(action)
        # quality = self.get_quality(self._t, from_state, action)
        return action_decoded.get("ax"), action_decoded.get("ay"), 0


    def set_policy(self, from_state, action):
        """
        Set policy for from_state to action
        """
        self._policy_map[from_state] = action


    def get_max_value_diff(self, t):
        """
        Get the max diff of values in the state set
        """
        max_diff = float("-inf")
        for from_state in self._states_set:
            diff = abs(
                self.get_value(t, from_state) - self.get_value(t-1, from_state))
            max_diff = max(max_diff, diff)
        return max_diff


    def set_quality(self, t, from_state, action):
        """
        Set the quality value
        """
        if not self._quality_map.get(t):
            self._quality_map[t] = {}
        qt = self._quality_map.get(t)
        if not qt.get(from_state):
            qt[from_state] = {}
        qts = qt.get(from_state)

        if (self.will_crash(from_state, action) or 
            self.will_stay(from_state, action)):
            # Set invalid states to negative infinity so they are the smallest
            qts[action] = float("-inf")
        else:
            qts[action] = self.get_quality(t, from_state, action)


    def get_quality(self, t, from_state, action):
        """
        Calculate the quality value
        """
        sigma = 0
        for to_state in self._states_set:
            sigma += (self.get_transition(from_state, action, to_state) * 
                    self.get_value(t-1, to_state))
        return self.reward(from_state, action) + self._gamma * sigma


    def get_max_quality(self, t, from_state):
        """
        Get the max quality for the state at given time.
        """
        max_qtsa = float("-inf")
        max_action = ""
        qt = self._quality_map.get(self._t)
        if qt is not None:
            qts = qt.get(from_state)
            if qts is not None:
                for action in qts:
                    qtsa = qts.get(action)
                    if qtsa > max_qtsa:
                        max_qtsa = qtsa
                        max_action = action
        return max_action, max_qtsa


    def init_values_map(self):
        """
        Initialize the value lookups by specifiying the V0 lookup.
        """
        for state in self._states_set:
            self.set_value(0, state, 0)

    def set_value(self, t, from_state, value):
        """
        Set value for the time and the state
        """
        v_lookup = self._values_map.get(t)
        if v_lookup is None:
            v_lookup = {}
            self._values_map[t] = v_lookup
        v_lookup[from_state] = value


    def get_value(self, t, from_state):
        """
        Get value for the time and the state
        """
        vt = self._values_map.get(t)
        if vt is not None:
            return vt.get(from_state) or 0
        return 0
    

    def init_transition_map(self):
        """
        Inititalize the transition map with all reachable state from all state
        """
        count = 0
        for from_state in self._states_set:
            count += 1
            self._transition_map[from_state] = {}
            from_decoded = self.decode_state(from_state)
            for encoded_action in self._actions_set:
                if self._simulator.track.isFinish(
                    from_decoded.get("x"), from_decoded.get("y")):
                    # Do not record transition for the finish state
                    continue

                if not self.will_crash(from_state, encoded_action):
                    to_state = self.get_cur_state()
                    self._transition_map[from_state][encoded_action] = to_state


    def get_transition(self, from_state, encoded_action, to_state):
        """
        Get the transition. Reachable to_state returns 1. Unreachable returns 0.
        """
        if (self._transition_map.get(from_state) and 
            self._transition_map.get(from_state).get(encoded_action) == to_state):
            return 1
        else:
            return 0
