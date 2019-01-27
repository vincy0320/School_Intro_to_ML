#!/usr/bin/python3
import random


FAIL_THRESHOLD = 0.2

class Simulator: 

    def __init__(self, track, car):
        """
        Constructor of a Track object.
        """
        self.track = track
        self.car = car
        self.__reset_car()
        self._reset_to_start = False

    def __reset_car(self, x = 0, y = 0):
        """
        Reset the car location
        """
        start_pos = random.choice(self.track.start_line())
        self.car.resetAt(start_pos[0], start_pos[1])


    def set_reset_to_start(self, should_reset_to_start):
        """
        Set the value for reset to start
        """
        self._reset_to_start = should_reset_to_start

    def is_reset_to_start(self):
        return self._reset_to_start


    def has_crashed(self):
        """
        Check if the car is crashed
        """
        carX, carY = self.car.get_position()
        return self.track.isWall(carX, carY)


    def has_finished(self):
        """
        Check if the car is at the finish line
        """
        carX, carY = self.car.get_position()
        return self.track.isFinish(carX, carY)


    def move(self, ax, ay, randomness = False):
        """
        Move a car on the track.
        When randomness is set to true, there is FAIL_THRESHOLD chance that the
        acceleration will not apply. 
        """
        if randomness and random.uniform(0, 1) < FAIL_THRESHOLD:
            # Simulate the requirement where 20% of the acceleration would fail
            # so that the speed would remain the same.
            ax = 0
            ay = 0
        self.car.accelerate(ax, ay)
        self.car.move()


    def simulate(self, x, y, vx, vy, ax, ay, randomness = False):
        """
        Set the state of the car
        When randomness is set to true, there is FAIL_THRESHOLD chance that the
        acceleration will not apply. 
        """
        self.car.set_state(x, y, vx, vy)
        self.move(ax, ay, randomness)


    def run(self, ax, ay, randomness = True):
        """
        Run the simulator with the given accelerations and randomnees. 
        When randomness is set to true, there is FAIL_THRESHOLD chance that the
        acceleration will not apply. 
        """
        if ax is None:
            self.reset_upon_crash()
            return

        self.move(ax, ay, randomness)
        if self.has_crashed():
            self.reset_upon_crash()

    def reset_upon_crash(self):
        if self._reset_to_start:
            self.reset()
        else:
            carX, carY = self.car.get_position()
            new_x, new_y = self.track.get_nearest_track_position(carX, carY)
            self.__reset_car(new_x, new_y)


    def step(self):
        """
        Move the simulator one step
        """
        ax_options, ay_options = self.car.get_acceleration_options()
        ax = random.choice(ax_options)
        ay = random.choice(ay_options)
        self.move(ax, ay, False)

        if self.has_crashed():
            self.__reset_car()
        return self.has_finished()


    def run_with_policy(self, policy, t):
        """
        Run the simulator with the given policy
        """
        if t == 0:
            self.__reset_car()
        
        states = policy.get(t)
        print(states)


    def reset(self):
        """
        Reset simulator to the initial point
        """
        self.__reset_car()


    def current_states(self):
        """
        Print the current state
        """

        track_copy = self.track.get_map()
        carX, carY = self.car.get_position()
        carVX, carVY = self.car.get_velocity()
        
        # Mark the car location
        track_copy[carY] = track_copy[carY][:carX] + "X" + track_copy[carY][carX+1:]
        
        track_states = ""
        for line in track_copy:
            track_states += line + "\n"

        # Print out the track and the car
        car_states = " ".join(["Car:",
            "Pos", str(carX), str(carY), 
            "Velocity", str(carVX), str(carVY)
        ])

        return [track_states, car_states]
    