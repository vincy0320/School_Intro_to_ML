#!/usr/bin/python3


class Track: 

    def __init__(self, input_file):
        """
        Constructor of a Track object.
        """

        self._track = []
        self._row_count = 0
        self._col_count = 0
        self._start_line = []
        self.__construct_track(input_file)
        self._input_file = input_file
        

    def __construct_track(self, input_file):
        """
        Construct track from the input file
        """

        with open(input_file) as f:
            content = f.readlines()

        splitted = content[0].split(",")
        self._row_count = int(splitted[0])
        self._col_count = int(splitted[1])
        
        row_index = 1
        for row_index in range(self._row_count):
            row = ""
            for col_index in range(self._col_count):
                state = content[1:][row_index][col_index]
                if state == "S":
                    self._start_line.append([col_index, row_index])
                row += state
            self._track.append(row)

    def __is_valud_coordinate(self, x, y):
        """
        Check whether the coordinate are valid
        """
        return 0 <= x < self._col_count and 0 <= y < self._row_count


    def isStart(self, x, y):
        """
        Check if the point is on the start line
        """
        return self.__is_valud_coordinate(x, y) and self._track[y][x] == "S"
            

    def isFinish(self, x, y):
        """
        Check if the point is on the finish line
        """
        return self.__is_valud_coordinate(x, y) and self._track[y][x] == "F"

    
    def isWall(self, x, y):
        """
        Check if the point is on the wall
        """
        return not self.__is_valud_coordinate(x, y) or self._track[y][x] == "#"

    def isOnTrack(self, x, y):
        """
        Check if the point is on the track (ie. not wall and finish)
        """
        return self.__is_valud_coordinate(x, y) and not self.isWall(x, y) and not self.isFinish(x, y)

    def start_line(self):
        """
        Return the start line coordinates
        """
        return self._start_line

    def get_map(self):
        """
        Get a copy of the map of the track
        """

        return self._track.copy()

    def get_nearest_track_position(self, x, y):
        """
        Get the nearest track position that's not a wall or a finish line.
        """
        step = 1
        while True:
            new_x = x + step
            if self.isOnTrack(new_x, y):
                return new_x, y
            new_x = x - step
            if self.isOnTrack(new_x, y):
                return new_x, y
            new_y = y + step
            if self.isOnTrack(x, new_y):
                return x, new_y
            new_y = y - step
            if self.isOnTrack(x, new_y):
                return x, new_y
            
            step += 1
            if (not self.__is_valud_coordinate(x + step, y) and
                not self.__is_valud_coordinate(x - step, y) and
                not self.__is_valud_coordinate(x, y + step) and
                not self.__is_valud_coordinate(x, y - step)):
                return -1, -1


    def print(self):
        """
        Print the track
        """
        for line in self._track:
            print(line)

    def get_name(self):
        """
        Get the name of the track. ie. the input file name.
        """
        return self._input_file