#!/usr/bin/python3

V_MAX = 5 # TODO: Change to 5
V_MIN = -5 # TODO: Change to -5
ACCELERATION = [-1, 0, 1]

class Car: 

    def __init__(self):
        """
        Constructor of a Track object.
        """
        # Position
        self._x = -1
        self._y = -1
        # Velocity
        self._vx = 0 
        self._vy = 0

    def __is_valid_v(self, v):
        """
        Check if the velocity is valid. ie. in range.
        """

        return V_MIN <= v <= V_MAX

    def set_state(self, x, y, vx, vy):
        """
        Set the state of the car
        """
        self._x = x
        self._y = y
        self._vx = vx
        self._vy = vy

    def is_state(self, x, y, vx, vy):
        """
        Check if the state of the car matches given params
        """
        return self._x == x and self._y == y and self._vx == vx and self._vy == vy

    def resetAt(self, x, y):
        """
        Set the start position of the car to x and y
        """
        self.set_state(x, y, 0, 0)

    def accelerate(self, ax, ay):
        """
        Set the acceleration of the car to x and y
        """
        new_vx = self._vx + ax
        new_vy = self._vy + ay
        if self.__is_valid_v(new_vx) and self.__is_valid_v(new_vy):
            self._vx = new_vx
            self._vy = new_vy
    

    def move(self):
        """
        Move the car one step from the current position at the current velocity
        """
        if self._x < 0 or self._y < 0:
            raise Exception("Error: Car has not been placed at the start yet")

        self._x += self._vx
        self._y += self._vy

    def get_position(self):
        """
        Get the current position of the car
        """
        return self._x, self._y

    def get_velocity(self):
        """
        Get the current velocity of the car
        """
        return self._vx, self._vy

    def get_velocity_range(self):
        """
        Get the range of velocity of the car in x y direction
        """
        return [
            list(range(V_MIN, V_MAX+1)),
            list(range(V_MIN, V_MAX+1))
        ]

    def get_acceleration_options(self):
        """
        Get the range of accelrations of the car in x y direction
        """
        return [ACCELERATION, ACCELERATION]
