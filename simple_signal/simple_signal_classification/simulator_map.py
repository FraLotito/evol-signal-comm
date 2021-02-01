"""
This file contains the implementation of a basic map for the simulator.

Created by Leonardo Lucio Custode on 8/11/2019
"""
import cv2
import time
import numpy as np
from entities import Void, Wall, PointOfInterest, Attacker, Defender, Neutral, PersonOfInterest


class Map:
    """
    This class represents a simple map for the simulator.
    """
    wall = -1
    void = 0
    attacker = 1
    defender = 2
    neutral = 3
    point_of_interest = 4
    person_of_interest = 5

    node2entity = {
        wall: Wall,
        void: Void,
        attacker: Attacker,
        defender: Defender,
        neutral: Neutral,
        point_of_interest: PointOfInterest,
        person_of_interest: PersonOfInterest,
    }

    @staticmethod
    def entity2node(e):
        """
        Inverse dictionary for node2entity.
        """
        for k, v in Map.node2entity.items():
            if isinstance(e, v):
                return k

    def __init__(self, **kwargs):
        """
        Initializes a new map

        A map is defined as a matrix whose elements can be:
        - void slots, represented by the number '0'
        - attackers, represented by the letter 'a'
        - defenders, represented by the letter 'd'
        - neutral agents, represented by the letter 'n'
        - points of interest, represented by the letter 'p'
        - people of interest, represented by the letter 'P'

        :param filepath: the path of the file containing the map
        :param matrix: the matrix containing the map
        :param attacker_agent: the agent to be assigned to the attacker
        :param defender_agent: the agent to be assigned to the attacker
        :param neutral_agent: the agent to be assigned to the attacker
        :param personoi_agent: the agent to be assigned to the attacker
        :param allowed_nodes: the nodes allowed in this map, a dict {symbol: class}
        """
        # Check whether the presence of two arguments returns True in a XOR
        if not (('matrix' in kwargs) ^ ('filepath' in kwargs)):
            raise AttributeError(
                "The constructor requires one and only one of the following keywords: matrix, filepath")

        # Load from file
        if 'filepath' in kwargs:
            matrix = self._load_from_file(kwargs['filepath'])
        else:
            matrix = kwargs['matrix']

        # self._ctr = {n: 0 for n in self.nodes.keys()}

        self._check_matrix(matrix)

        # Decode the matrix
        self._decode_matrix(matrix)

        # Assign the agents to the map

    def _decode_matrix(self, matrix):
        self._status = list(matrix)
        for i in range(len(self._status)):
            self._status[i] = np.array([k for k in self._status[i]], dtype=np.int8)
        self._status = np.array(self._status)
        self.height, self.width = self._status.shape

        # for j in range(self._status.shape[0]):
        #     for i in range(self._status.shape[1]):
        #         # Get the type of node
        #         nodeval = self._status[j, i]
        #
        #         # Create an instance of that node and assign it its unique id (within the category)
        #         self._status[j, i] = self.nodes[nodeval](id=self._ctr[nodeval])
        #
        #         # Increase the id counter
        #         self._ctr[nodeval] += 1

    def _check_matrix(self, matrix):
        valid = [n for n in self.node2entity.keys()]
        l0 = len(matrix[0])
        if l0 == 0:
            raise ValueError('Matrix is not well formed.\nEach row must contain at least one element.')
        for m in matrix:
            if len(m) != l0:
                raise ValueError('Matrix is not well formed.\nExpected len: {}, actual len: {}.'.format(l0, len(m)))
            for c in m:
                if c not in valid:
                    raise ValueError('The matrix contains invalid data.\nValid ids: {}.\nFound: {}.'.format(valid, c))

    @staticmethod
    def _load_from_file(filepath):
        return np.loadtxt(filepath, dtype=np.int8)

    def __getitem__(self, item):
        return self._status[tuple(item)]

    def __setitem__(self, position, item):
        self._status[tuple(position)] = item

    def plot(self, time_delay=1000):
        """
        Plots the map and displays it in a GUI.
        :param time_delay: the delay between two consecutive frames.
        :returns True if the "Esc" key has been pressed, False otherwise
        """
        scale = 10
        repr = np.ones((self.height * scale, self.width * scale, 3), dtype=np.uint8) * 255
        for j in range(self.height):
            for i in range(self.width):
                self.node2entity[self._status[j, i]](None).get_graphical_object(repr, (i * scale, j * scale),
                                                                                ((i + 1) * scale, (j + 1) * scale))
        cv2.imshow("simulation", repr)
        k = cv2.waitKey(time_delay)
        if k == 27:  # Check if the "Esc" key has been pressed
            return True
        return False

    def close_plot(self):
        """
        Closes the plot and destroys the window.
        """
        time.sleep(2)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    filepath = "tests/files/10_wls_map_20x20.map"
    m = Map(filepath=filepath)
    m.plot()
