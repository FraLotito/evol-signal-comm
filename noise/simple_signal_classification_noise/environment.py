"""
This file contains the implementation of a basic environment for the simulator.
New environments (i.e. with different "laws") can be easily created by extending this class and overriding the needed
methods.

Created by Leonardo Lucio Custode on 14/11/2019
"""
import numpy as np
from simulator_map import Map
from entities import Neutral, Defender, Attacker, PersonOfInterest, Wall, Void, PointOfInterest, Entity, Agent


class Environment:
    """
    A basic implementation of an environment.
    """

    class _EntityState:
        """
        An utility class that wraps an entity and its position into the map
        """

        def __init__(self, entity, position):
            self.entity = entity
            self.position = position

    # Definition of the possible outcomes of the simulation
    tie = -1  # In case the time needed exceeds the maximum time allowed for the simulation
    attackers_victory = 0
    defenders_victory = 1

    def __init__(self, sim_map, show=False, max_timesteps: int = 100, line_of_sight: int = 3):
        """
        Initializes a new Environment.
        :param sim_map: The map to use for the simulation. It can also be a blank map (np.zeros).
        :param show: Boolean flag indicating whether the simulation must be shown to the user or not
        :param max_timesteps: integer parameter defining the maximum length of a simulation
        :param line_of_sight: integer representing the radius of the circle inscribed into the square representing the
                                receptive field
        """
        self.max_timesteps = max_timesteps
        self.timestep = 0
        self.show_simulation = show

        self._time_delay = 100
        self._next_id = 0
        self._map = Map(matrix=sim_map)
        self._entities = {}
        self._attackers = {}
        self._defenders = {}
        self._neutrals = {}
        self._peopleoi = {}
        self._pois = {}
        self._walls = {}
        self._line_of_sight = line_of_sight
        self.res = None
        self.close = False

    def addWall(self, id: int, position: tuple):
        """
        Adds a new wall to the environment.
        :param id: the id of the wall
        :param position: a tuple (y, x) specifying the location of the wall
        """
        self.add(Wall(id), position)

    def addPOI(self, id: int, position: tuple):
        """
        Adds a new point of interest to the environment.
        :param id: the id of the point of interest
        :param position: a tuple (y, x) specifying the location of the point of interest
        """
        self.add(PointOfInterest(id), position)

    def getNextId(self):
        """
        Returns the first free id.
        """
        for i in range(len(self._entities), -1, -1):
            if i not in self._entities:
                return i
        raise ValueError("Something wrong happened. There are more ids than entities")

    def addPersonOfInterest(self, agent: PersonOfInterest, position: tuple):
        """
        Adds a new person of interest to the environment.
        :param id: the id of the person of interest
        :param position: a tuple (y, x) specifying the location of the person of interest
        """
        self._check_type(agent, PersonOfInterest)
        self.add(agent, position)

    def addNeutral(self, agent: Neutral, position: tuple):
        """
        Adds a new neutral agent to the environment.
        :param id: the id of the neutral agent
        :param position: a tuple (y, x) specifying the location of the neutral agent
        """
        self._check_type(agent, Neutral)
        self.add(agent, position)

    def addDefender(self, agent: Defender, position: tuple):
        """
        Adds a new defender to the environment.
        :param id: the id of the defender
        :param position: a tuple (y, x) specifying the location of the defender
        """
        self._check_type(agent, Defender)
        self.add(agent, position)

    def addAttacker(self, agent: Attacker, position: tuple):
        """
        Adds a new attacker to the environment.
        :param id: the id of the attacker
        :param position: a tuple (y, x) specifying the location of the attacker
        """
        self._check_type(agent, Attacker)
        self.add(agent, position)

    def getLocation(self, id: int):
        """
        Returns the location of the entity with the passed id.
        :param id: the id of the entity
        """
        return self._entities[id].position

    def getAllEntities(self):
        """
        Returns all the entities of the environment.
        """
        return [a.entity for a in self._entities.values()]

    def getEntities(self, entity_type):
        """
        Returns all the entities of a given type
        """
        return [a.entity for a in self._entities.values() if isinstance(a.entity, entity_type)]

    def getAgents(self):
        """
        Returns all the agents.
        """
        return self.getEntities(Agent)

    def getAttackers(self):
        """
        Returns all the attackers.
        """
        return self.getEntities(Attacker)

    def getDefenders(self):
        """
        Returns all the defenders.
        """
        return self.getEntities(Defender)

    def getNeutrals(self):
        """
        Returns all the neutral agents
        """
        return self.getEntities(Neutral)

    def getPeopleOfInterest(self):
        """
        Returns all the people of interest.
        """
        return self.getEntities(PersonOfInterest)

    def getPOIs(self):
        """
        Returns all the points of interest.
        """
        return self.getEntities(PointOfInterest)

    def getWalls(self):
        """
        Returns all the walls.
        """
        return self.getEntities(Wall)

    def getAgent(self, id: int):
        """
        Returns the agent associated to the passed id.
        """
        return self._entities[id].entity

    def _check_position(self, position):
        """
        Checks if a given position is free and correct.
        """
        if not (0 <= position[0] < self._map.height) or not (0 <= position[1] < self._map.width):
            raise ValueError("The passed point lies outside the map")
        if not self.is_free(position):
            raise ValueError("Trying to add an entity to a non-free position.")

    def add(self, entity: Entity, position: tuple):
        """
        Adds a new entity into the environment.
        :param entity: the entity to add
        :param position: tuple (y, x) where the entity has to be added
        """
        self._check_position(position)
        self._check_id(entity.id)
        self._entities[entity.id] = self._EntityState(entity, position)

        if isinstance(entity, Attacker):
            self._attackers[entity.id] = self._entities[entity.id]
        elif isinstance(entity, Defender):
            self._defenders[entity.id] = self._entities[entity.id]
        elif isinstance(entity, Neutral):
            self._neutrals[entity.id] = self._entities[entity.id]
        elif isinstance(entity, PersonOfInterest):
            self._peopleoi[entity.id] = self._entities[entity.id]
        elif isinstance(entity, Wall):
            self._walls[entity.id] = self._entities[entity.id]
        elif isinstance(entity, PointOfInterest):
            self._pois[entity.id] = self._entities[entity.id]

        self._map[list(position)] = Map.entity2node(entity)

    def is_free(self, position):
        """
        Returns True if the passed position is free.
        :param position: tuple (y, x)
        """
        return self._map[position[0], position[1]] == Map.void

    def _check_id(self, id: int):
        """
        Checks whether the passed id already exists. If so, raises a ValueError.
        :param id: the id to check
        """
        if id in self._entities:
            raise ValueError("Trying to add an entity with a non-free id.")

    def _check_type(self, agent, type):
        """
        Checks whether the passed agent corresponds to the passed type. If it does not, raises a ValueError.
        :param agent: the agent to check
        :param type: the class of agents
        """
        if not isinstance(agent, type):
            raise ValueError("The passed object does not correspond to an instance of the type of agent being added.")

    def remove(self, id):
        """
        Removes the entity associated to the given id from the environment.
        :param id: the id of the entity to remove.
        """
        position = self._entities[id].position
        self._map[position] = Map.void
        a = self._entities[id].entity

        if isinstance(a, Attacker):
            del self._attackers[id]
        elif isinstance(a, Defender):
            del self._defenders[id]
        elif isinstance(a, Neutral):
            del self._neutrals[id]
        elif isinstance(a, PersonOfInterest):
            del self._peopleoi[id]
        elif isinstance(a, PointOfInterest):
            del self._pois[id]
        elif isinstance(a, Wall):
            del self._walls[id]
        del self._entities[id]

    def executeTimestep(self):
        """
        Executes a timestep in the simulation of the environment.
        """
        if not self.game_ended() and not self.close:
            self.res = self._compute_next_status()

            if self.res == self.attackers_victory:
                return self.attackers_victory
            elif self.res == self.defenders_victory:
                return self.defenders_victory

            if self.show_simulation:
                self.close = self._map.plot(self._time_delay)
            if not self.close:
                self.close = (self.timestep >= self.max_timesteps - 1)
            self.timestep += 1
        else:
            self._map.close_plot()
            if self.res is None:
                self.res = self.tie
            return self.res

    def getByLocation(self, target_pos):
        """
        Returns the entity that lies into the target position.
        :param target_pos: tuple (y, x)
        :returns the object lying in that position. None if the position is void.
        """
        obj = self._map[target_pos]
        if obj == Map.void:
            return None
        if obj == Map.wall:
            return self._search_for_position(self._walls, target_pos)
        if obj == Map.attacker:
            return self._search_for_position(self._attackers, target_pos)
        if obj == Map.defender:
            return self._search_for_position(self._defenders, target_pos)
        if obj == Map.neutral:
            return self._search_for_position(self._neutrals, target_pos)
        if obj == Map.person_of_interest:
            return self._search_for_position(self._peopleoi, target_pos)
        if obj == Map.point_of_interest:
            return self._search_for_position(self._pois, target_pos)

    def _compute_next_status(self):
        agents_actions = self._get_actions()

        # Gets rid of things like clashing agents or hitting a wall
        agents_actions = self._manage_conflicting_actions(agents_actions)

        # Here I give priority to the attackers for deciding who won. I do this because the goal of the system is to
        #   minimize the winning probability of attackers. For this reason this is going to be more robust.
        # Note: the priority is given only in the case that there is a single attacker remaining on a goal point while
        #   a defender is near him (not clashing, in case of clash it is captured)
        if self.detect_attackers_victory(agents_actions):
            return self.attackers_victory

        # Manage captures of attackers by the defenders
        captured, agents_actions = self._detect_captures(agents_actions)

        # Update status
        self._update_status(agents_actions)

        # Manage victory of the defenders
        if self.detect_defenders_victory(captured):
            return self.defenders_victory

        # Free some memory
        del agents_actions
        return None

    @staticmethod
    def _search_for_position(entity_dict, target_pos):
        for k, v in entity_dict.items():
            if v.position == target_pos:
                return v.entity
        return None

    def _get_actions(self):
        agents_actions = {}
        for entity in self._entities.values():
            e = entity.entity
            if isinstance(e, Agent):
                e.feedInput(self._get_representation_state(entity))
                action = e.getOutput()

                new_position = self._get_new_position(entity.position, action)
                agents_actions[e] = (entity.position, new_position)

        return agents_actions

    def _manage_conflicting_actions(self, agents_actions):
        conflicting_locations = self._find_conflicting_locations(agents_actions)
        found_defender = False
        for position, conflicts in conflicting_locations.items():
            defender_flag = conflicts[0]
            for e in conflicts[1]:
                if defender_flag and isinstance(e, Attacker):
                    agents_actions[e] = agents_actions[e][0], None
                else:
                    if not (isinstance(e, Attacker) and (self._map[agents_actions[e][1]] == Map.point_of_interest or
                                                         self._map[agents_actions[e][1]] == Map.person_of_interest)):
                        if isinstance(e, Defender) and not found_defender:
                            e.captured_attackers += 1
                            found_defender = True
                        agents_actions[e] = agents_actions[e][0], agents_actions[e][0]  # Bounce

        return agents_actions

    def _get_new_position(self, position, action):
        new_pos = list(position)
        if action == Agent.moveUp:
            new_pos[0] -= 1
        elif action == Agent.moveDown:
            new_pos[0] += 1
        elif action == Agent.moveLeft:
            new_pos[1] -= 1
        elif action == Agent.moveRight:
            new_pos[1] += 1

        if (not self._is_in_map(new_pos)) or (
                not self.is_free(new_pos) and not (self._map[position] == Map.attacker and (
                self._map[new_pos] == Map.point_of_interest or self._map[
            new_pos] == Map.point_of_interest))):
            return position
        return tuple(new_pos)

    @staticmethod
    def _find_conflicting_locations(agents_actions):
        """
        :param agents_actions:
        :return: {conflicting_location: (there_is_defender, agents_colliding)}
        """
        next_positions = {}
        for a, (_, next) in agents_actions.items():
            if next not in next_positions:
                next_positions[next] = [False, []]
            next_positions[next][1].append(a)
            if isinstance(a, Defender):
                next_positions[next][0] = True
        conflicting = {k: v for k, v in next_positions.items() if len(v[1]) > 1}
        return conflicting

    def detect_attackers_victory(self, agents_actions):
        """
        Detects whether the next state of the environment is in a state which determines the victory of the attackers.
        :param agents_actions: a dictionary {agent: (prev_pos, next_pos)} describing the next moves for each agent.
        """
        for agent, (_, next_loc) in agents_actions.items():
            if isinstance(agent, Attacker):
                if self._map[next_loc] == Map.point_of_interest or self._map[next_loc] == Map.person_of_interest:
                    return True

    def detect_defenders_victory(self, captured):
        """
        Detects whether the next state of the environment is in a state which determines the victory of the defenders.
        :param captured: an integer indicating how many attackers have been captured in the current timestep.
        """
        return len(self._attackers) == 0 or len(self._attackers) == captured

    @staticmethod
    def _detect_captures(agents_actions):
        attackers = []
        attacker_positions = []
        defenders = []
        defender_positions = []

        for a, (prev, _) in agents_actions.items():
            if isinstance(a, Attacker):
                attackers.append(a)
                attacker_positions.append(prev)
            elif isinstance(a, Defender):
                defenders.append(a)
                defender_positions.append(prev)

        a_x = np.array([a[1] for a in attacker_positions])
        a_y = np.array([a[0] for a in attacker_positions])
        d_x = np.array([a[1] for a in defender_positions])
        d_y = np.array([a[0] for a in defender_positions])

        dist_x = np.zeros((len(defenders), len(attackers)))
        dist_y = np.zeros((len(defenders), len(attackers)))

        for j in range(len(defenders)):
            for i in range(len(attackers)):
                dist_x[j, i] = abs(a_x[i] - d_x[j])
                dist_y[j, i] = abs(a_y[i] - d_y[j])

        dist = dist_x + dist_y
        i_y, i_x = np.where(dist == 1)
        captured = 0

        for j, i in zip(i_y, i_x):
            if agents_actions[attackers[i]] is not None:
                agents_actions[attackers[i]] = agents_actions[attackers[i]][0], None
                defenders[j].captured_attackers += 1
                captured += 1

        return captured, agents_actions

    def _update_status(self, agents_actions):
        for a, (prev, next) in agents_actions.items():
            if next is None:
                if isinstance(a, Attacker):
                    del self._entities[a.id]
                    del self._attackers[a.id]
                    self._map[prev] = Map.void
            else:
                self.moveAgent(a.id, src=prev, dst=next)

    def moveAgent(self, id, src, dst):
        """
        Moves the agent from the src position to the dest position.
        :param id: the id of the agent to move
        :param src: the source position (y, x) where the agent comes from
        :param dst: the destination (y, x) where the agent needs to move
        """
        symbol = None
        to_update = None

        a = self._entities[id].entity

        if isinstance(a, Attacker):
            to_update = self._attackers
            symbol = Map.attacker
        if isinstance(a, Defender):
            to_update = self._defenders
            symbol = Map.defender
        if isinstance(a, Neutral):
            to_update = self._neutrals
            symbol = Map.neutral
        if isinstance(a, PersonOfInterest):
            to_update = self._peopleoi
            symbol = Map.person_of_interest

        if symbol and to_update and self._is_in_map(dst) and self.is_free(dst):
            self._entities[id].position = dst
            to_update[id].position = dst

            self._map[src] = Map.void
            self._map[dst] = symbol

    def _get_representation_state(self, entity):
        pos = entity.position

        xm = max(0, pos[1] - self._line_of_sight)
        xM = min(self._map.width, pos[1] + self._line_of_sight + 1)
        ym = max(0, pos[0] - self._line_of_sight)
        yM = min(self._map.height, pos[0] + self._line_of_sight + 1)

        return self._map[ym:yM, xm:xM]

    def moveAgentTo(self, id, direction):
        """
        Moves the agent one step int the given direction.
        :param id: the id of the agent to move
        :param direction: the direction where the agent has to move
        """
        if direction == Agent.moveDown:
            coord = 0
            coef = 1
        if direction == Agent.moveUp:
            coord = 0
            coef = -1
        if direction == Agent.moveRight:
            coord = 1
            coef = 1
        if direction == Agent.moveLeft:
            coord = 1
            coef = -1

        old_pos = self._entities[id].position

        new_pos = list(old_pos)
        new_pos[coord] += coef
        new_pos = tuple(new_pos)

        self.moveAgent(id, old_pos, new_pos)

    def game_ended(self):
        """
        Returns True if the simulation is finished False, otherwise
        """
        return self.res is not None

    def _is_in_map(self, dst):
        return (0 <= dst[0] < self._map.height) and (0 <= dst[1] < self._map.width)


if __name__ == '__main__':
    class DummyAttacker(Attacker):
        def feedInput(self, state):
            pass

        def getOutput(self):
            return Agent.moveRight


    sim_map = np.zeros((20, 20), dtype=np.int8)
    e = Environment(sim_map, show=True, max_timesteps=20)
    e.addPOI(0, (10, 10))

    a = DummyAttacker(1)
    e.addAttacker(a, (10, 1))

    e.addWall(2, (10, 8))
    e.addWall(3, (9, 8))
    e.addWall(4, (11, 8))

    while not e.game_ended():
        e.executeTimestep()

    print("Result of the simulation: {}".format(e.res))
