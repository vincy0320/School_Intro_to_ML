#!/usr/bin/python3
import random
import rl_base

SEPARATOR = "_"
DISCOUNT_RATE = 0.7


class ReinformacementLearning(rl_base.RLBase): 

    def __init__(self, simulator, epsiodes = 100):
        """
        Constructor of a model based reinforcement learning
        """
        rl_base.RLBase.__init__(self, simulator)
        self._learning_rate = 0.7
        self._epsilon = 0.2
        self._episodes = epsiodes
        # quality_map is the q-function from one state and an action to a 
        # real number
        self._quality_table = {}
        self.init_quality_table()


    def get_attributes(self):
        """
        Get the attribute of model
        """
        return {
            "Learning Rate": self._learning_rate,
            "Epsilon Greedy": self._epsilon,
            "Episodes": self._episodes
        }

    def q_learn(self):
        """
        Perform Q Learning
        """
        ep = 0
        while ep < self._episodes:
            # get initial state
            self._simulator.reset()
            from_state = self.get_cur_state()
            while not self.isFinished(from_state):
                action = self.choose_action(from_state)[0] # Q-Learn specific
                if action is None:
                    # No action can be chosen, then skip this from_state
                    break
                self.simulate(from_state, action)
                new_from_state = self.get_cur_state()
                # Calculate the quality
                updated_quality, new_action = self.get_updated_quality(
                    from_state, action, new_from_state)
                updated = self.set_quality(from_state, action, updated_quality)
                if updated:
                    from_state = new_from_state
                    action = new_action
            ep += 1
        return


    def sarsa_learn(self):
        """
        """
        ep = 0
        while ep < self._episodes:
            # get initial state
            self._simulator.reset()
            from_state = self.get_cur_state()
            action = self.choose_action(from_state)[0] # SARSA Specific
            if action is None:
                # No action can be chosen, then skip this from_state
                continue
            while not self.isFinished(from_state):
                self.simulate(from_state, action)
                new_from_state = self.get_cur_state()
                if not self.is_valid_state(new_from_state):
                    break
                new_action = self.choose_action(new_from_state)[0] # SARSA Specific
                if new_action is None:
                    # No action can be chosen, then skip this from_state action
                    # combination
                    break
                # Calculate the quality
                updated_quality = self.get_updated_quality(
                    from_state, action, new_from_state, new_action)[0]
                updated = self.set_quality(from_state, action, updated_quality)
                if updated:
                    from_state = new_from_state
                    action = new_action
                else:
                    # If the greedy choice didn't work, 
                    break
            ep += 1
        return 


    def get_action(self):
        """
        Get an action to execute from the curren state
        """
        x, y = self._simulator.car.get_position()
        vx, vy = self._simulator.car.get_velocity()
        from_state = self.encode_state(x, y, vx, vy)
        action, quality = self.get_max_quality(from_state)
        if action is None:
            return None, None, float("-inf")
        action_decoded = self.decode_action(action)
        return action_decoded.get("ax"), action_decoded.get("ay"), quality

    def is_valid_state(self, state):
        return state in self._states_set


    def isFinished(self, encoded_state):
        """
        Checks whether a encoded state is finished
        """
        decoded_state = self.decode_state(encoded_state)
        return self._simulator.track.isFinish(
                    decoded_state.get("x"), decoded_state.get("y"))


    def init_quality_table(self):
        """
        Initialize the Quality table with random value selected from natural 
        distribution
        """
        for from_state in self._states_set:
            for encoded_action in self._actions_set:
                value = random.uniform(0, 1)
                self.set_quality(from_state, encoded_action, value)



    def set_quality(self, from_state, action, value):
        """
        Set the quality table value for given state and action, skip the invalid
        combinations
        """
        if (self.will_crash(from_state, action) or 
            self.will_stay(from_state, action)):
            # Don't record bad state-action combinations
            return False

        if not value > float("-inf"):
            # Don't record neg inf values
            return False

        if not self._quality_table.get(from_state):
            self._quality_table[from_state] = {}
        q_table_row = self._quality_table.get(from_state)
        q_table_row[action] = value
        return True


    def get_updated_quality(self, from_state, action, to_state, new_action = None):
        """
        Calculate and return the updated quality value.
        """
        updated_quality = float("-inf")
        q_table_row = self._quality_table.get(from_state)
        if q_table_row is None:
            # Invalid from state
            return float("-inf"), new_action

        cur_quality = q_table_row.get(action)
        if cur_quality is None:
            # Invalid action
            return float("-inf"), new_action

        q_table_row = self._quality_table.get(to_state)
        if q_table_row is None:
            # Invalid to state
            return float("-inf"), new_action

        next_quality = 0
        if new_action is None:
            # Q learning doesn't specify a new action
            new_action, next_quality = self.get_max_quality(to_state)
            if new_action is None:
                # No available action at the to state
                return float("-inf"), new_action
        else:
            # SARSA
            next_quality = q_table_row.get(new_action)
            if next_quality is None:
                # The given new action is not valid
                return float("-inf"), new_action

        if cur_quality > float("-inf"):
            # Calculate the new value only if the cur_quality is valid
            updated_quality = cur_quality + self._learning_rate * (
                self.reward(from_state, action) + self._gamma * next_quality 
                - cur_quality)
        return updated_quality, new_action


    def get_max_quality(self, from_state):
        """
        Get the action and the quality itself that maximizes quality at the
        from state
        """
        max_quality = float("-inf")
        max_action = None
        q_table_row = self._quality_table.get(from_state)
        if q_table_row is not None:
            for action in q_table_row:
                action_quality = q_table_row.get(action)
                if action_quality > max_quality:
                    max_quality = action_quality
                    max_action = action
        return max_action, max_quality

    
    def choose_action(self, from_state):
        """
        Choose action using epsilon greedy algorithm
        """
        prob = random.uniform(0, 1)
        max_q_action = None
        random_action_tried = set()
        is_greedy = True
        while max_q_action is None:
            # Choose a random action with probabily of epsilon
            if prob < self._epsilon:
                max_q_action = random.choice(list(self._actions_set))
                if (self.will_crash(from_state, max_q_action) or 
                    self.will_stay(from_state, max_q_action)):
                    random_action_tried.add(max_q_action)
                    if len(random_action_tried) == len(self._actions_set):
                        # exhausted all action options, give up
                        break
                    else:
                        continue
                is_greedy = False
            else:
                # Choose an action that maximizes the quality at the current state
                max_q_action = self.get_max_quality(from_state)[0]
                if max_q_action is None:
                    break
        return max_q_action, is_greedy


