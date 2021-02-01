"""
This file contains the implementation of all the basic entities for the simulator.
An entity is a dummy object which can be present in the map.
Agents are also entities, but they can also sense and perform action on the environment.

Created by Leonardo Lucio Custode on 8/11/2019
"""
import cv2


class Entity:
    """A dummy object that can exist in a map."""

    _color = (255, 255, 255)
    """Represents the color of the corresponding object drawn in the map"""

    def __init__(self, id: int):
        """
        Initializes a new entity
        :param id: uniquely identifies the entity in the current environment
        """
        self.id = id

    def get_graphical_object(self, repr, p0, p1):
        """
        Draws the graphical object associated to the entity into the given map.
        For generic entities the associated symbol is a rectangle.
        :param repr: matrix representation of the map
        :param p0: (x0, y0) associated to the left-top position of the entity into the matrix (repr)
        :param p1: (x1, y1) opposite to p0 associated to the right-bottom position of the entity into the matrix (repr)
        """
        cv2.rectangle(repr, p0, p1, self._color, -1)


class Void(Entity):
    """Defines a void entity. Useful for changing the background color."""
    _color = (255, 255, 255)


class Wall(Entity):
    """Defines a wall. Useful for changing walls color."""
    _color = (0, 0, 0)


class PointOfInterest(Entity):
    """Defines a POI. Useful for changing POI color."""
    _color = (0, 255, 0)


class Agent(Entity):
    """Defines a general agent and its actions."""

    stay = 0
    moveLeft = 1
    moveRight = 2
    moveUp = 3
    moveDown = 4
    capture = 5

    def __init__(self, id):
        super().__init__(id)

    @staticmethod
    def _mean_point(p0, p1, i):
        """
        Calculates the mean point of two vectors along an axis i. Used for graphical purposes.
        :param p0: The initial point
        :param p1: The final point
        :param i: the axis
        :returns the mean point along the axis i
        """
        return (p0[i] + p1[i]) // 2

    @staticmethod
    def _circle_position(p0, p1):
        """
        Returns the coordinates of the center of a circle given by the mean point between two vectors.
        :param p0: the initial point
        :param p1: the final point
        :returns the coordinates of the center of the circle
        """
        return Agent._mean_point(p0, p1, 0), Agent._mean_point(p0, p1, 1)

    def get_graphical_object(self, repr, p0, p1):
        """
        Draws the graphical object associated to the entity into the given map.
        For generic agents the associated symbol is a circle.
        :param repr: matrix representation of the map
        :param p0: (x0, y0) associated to the left-top position of the agent into the matrix (repr)
        :param p1: (x1, y1) opposite to p0 associated to the right-bottom position of the agent into the matrix (repr)
        """
        cv2.circle(repr, Agent._circle_position(p0, p1), (p1[0] - p0[0]) // 2, self._color, -1)

    def feedInput(self, state):
        """
        Feeds the current state into the "receptive field" of the agent.
        :param state: the state passed to the agent
        """
        raise NotImplementedError("This method must be implemented by the extending class")

    def getOutput(self):
        """
        Retrieves the next action that has to be performed by the current agent.
        """
        raise NotImplementedError("This method must be implemented by the extending class")


class Attacker(Agent):
    """Defines a generic attacker"""
    _color = (255, 0, 0)


class Defender(Agent):
    """Defines a generic defender"""
    _color = (0, 0, 255)

    def __init__(self, id):
        super().__init__(id)
        self.captured_attackers = 0


class Neutral(Agent):
    """Defines a generic neutral agent"""
    _color = (125, 125, 125)


class PersonOfInterest(Agent):
    """Defines a generic person of interest"""
    _color = (0, 125, 0)


class PerfectDefender(Defender):
    """
    A defender that captures an attacker whenever it is in its field of capture
    """
    pass


class ImperfectDefender(Defender):
    """
    A defender that captures an attacker in its field of capture accordingly to a capture probability
    """
    pass
