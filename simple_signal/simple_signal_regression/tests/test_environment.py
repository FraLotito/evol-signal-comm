import sys

from simulator_map import Map

sys.path.append(".")
sys.path.append("..")

import random
import pytest
import numpy as np
from functools import partial

from environment import Environment
from entities import Attacker, Defender, Neutral, PersonOfInterest, PointOfInterest, Wall, Agent


def create_blank_environment():
    sim_map = np.zeros((100, 100), dtype=np.int8)
    return Environment(sim_map)


def check_res_len(var, value):
    res = var()
    assert len(res) == value, "Expected {0} {1}. Got {2}.".format(value,
                                                                  var.__name__.lower().replace("get", ""),
                                                                  len(res))


def check_env(e, attackers, defenders, neutrals, people_of_interest, pois, walls):
    exp_agents = attackers + defenders + neutrals + people_of_interest
    check_res_len(e.getAgents, exp_agents)
    check_res_len(e.getAttackers, attackers)
    check_res_len(e.getDefenders, defenders)
    check_res_len(e.getNeutrals, neutrals)
    check_res_len(e.getPeopleOfInterest, people_of_interest)
    check_res_len(e.getPOIs, pois)
    check_res_len(e.getWalls, walls)


def test_blank_env():
    e = create_blank_environment()
    check_env(
        e,
        attackers=0,
        defenders=0,
        neutrals=0,
        people_of_interest=0,
        pois=0,
        walls=0,
    )


def check_agent_pos(e, id, position):
    assert e.getLocation(id) == position, "The location of the agent is not correct."


def test_valid_attacker_creation():
    e = create_blank_environment()
    e.addAttacker(Attacker(0), (10, 20))
    check_env(
        e,
        attackers=1,
        defenders=0,
        neutrals=0,
        people_of_interest=0,
        pois=0,
        walls=0,
    )
    check_agent_pos(e, 0, (10, 20))


def test_inv_pos_attacker_creation():
    e = create_blank_environment()
    try:
        e.addAttacker(Attacker(0), (120, 20))
        raise NotImplementedError("The agent has been added outside the sim_map without raising an exception.")
    except ValueError:
        pass


def test_inv_class_attacker_creation():
    e = create_blank_environment()
    try:
        e.addAttacker(Defender(0), (12, 20))
        raise NotImplementedError("A Defender has been added as Attacker without raising an exception.")
    except ValueError:
        pass


def test_two_attacker_creation():
    e = create_blank_environment()
    e.addAttacker(Attacker(0), (20, 20))
    e.addAttacker(Attacker(1), (40, 40))
    check_agent_pos(e, 0, (20, 20))
    check_agent_pos(e, 1, (40, 40))
    check_env(
        e,
        attackers=2,
        defenders=0,
        neutrals=0,
        people_of_interest=0,
        pois=0,
        walls=0,
    )


def test_autoid_attacker_creation():
    e = create_blank_environment()
    e.addAttacker(Attacker(0), (20, 20))
    e.addAttacker(Attacker(e.getNextId()), (40, 40))
    assert isinstance(e.getAgent(0), Attacker), "The agent added does not correspond to an Attacker"
    check_agent_pos(e, 0, (20, 20))
    check_agent_pos(e, 1, (40, 40))
    check_env(
        e,
        attackers=2,
        defenders=0,
        neutrals=0,
        people_of_interest=0,
        pois=0,
        walls=0,
    )


def test_inv_id_attacker_creation():
    e = create_blank_environment()
    e.addAttacker(Attacker(0), (20, 20))
    try:
        e.addAttacker(Attacker(0), (40, 40))
        raise NotImplementedError("An agent has been added with an existing id without raising an exception.")
    except ValueError:
        pass


def test_valid_defender_creation():
    e = create_blank_environment()
    e.addDefender(Defender(0), (10, 20))
    check_env(
        e,
        defenders=1,
        attackers=0,
        neutrals=0,
        people_of_interest=0,
        pois=0,
        walls=0,
    )
    check_agent_pos(e, 0, (10, 20))


def test_inv_pos_defender_creation():
    e = create_blank_environment()
    try:
        e.addDefender(Defender(0), (120, 20))
        raise NotImplementedError("The agent has been added outside the sim_map without raising an exception.")
    except ValueError:
        pass


def test_inv_class_defender_creation():
    e = create_blank_environment()
    try:
        e.addDefender(Attacker(0), (12, 20))
        raise NotImplementedError("A Attacker has been added as Defender without raising an exception.")
    except ValueError:
        pass


def test_two_defender_creation():
    e = create_blank_environment()
    e.addDefender(Defender(0), (20, 20))
    e.addDefender(Defender(1), (40, 40))
    check_agent_pos(e, 0, (20, 20))
    check_agent_pos(e, 1, (40, 40))
    check_env(
        e,
        defenders=2,
        attackers=0,
        neutrals=0,
        people_of_interest=0,
        pois=0,
        walls=0,
    )


def test_autoid_defender_creation():
    e = create_blank_environment()
    e.addDefender(Defender(0), (20, 20))
    e.addDefender(Defender(e.getNextId()), (40, 40))
    check_agent_pos(e, 0, (20, 20))
    check_agent_pos(e, 1, (40, 40))
    check_env(
        e,
        defenders=2,
        attackers=0,
        neutrals=0,
        people_of_interest=0,
        pois=0,
        walls=0,
    )


def test_inv_id_defender_creation():
    e = create_blank_environment()
    e.addDefender(Defender(0), (20, 20))
    try:
        e.addDefender(Defender(0), (40, 40))
        raise NotImplementedError("An agent has been added with an existing id without raising an exception.")
    except ValueError:
        pass


def test_valid_neutral_creation():
    e = create_blank_environment()
    e.addNeutral(Neutral(0), (10, 20))
    check_env(
        e,
        neutrals=1,
        attackers=0,
        defenders=0,
        people_of_interest=0,
        pois=0,
        walls=0,
    )
    check_agent_pos(e, 0, (10, 20))


def test_inv_pos_neutral_creation():
    e = create_blank_environment()
    try:
        e.addNeutral(Neutral(0), (120, 20))
        raise NotImplementedError("The agent has been added outside the sim_map without raising an exception.")
    except ValueError:
        pass


def test_inv_class_neutral_creation():
    e = create_blank_environment()
    try:
        e.addNeutral(Attacker(0), (12, 20))
        raise NotImplementedError("A Attacker has been added as Neutral without raising an exception.")
    except ValueError:
        pass


def test_two_neutral_creation():
    e = create_blank_environment()
    e.addNeutral(Neutral(0), (20, 20))
    e.addNeutral(Neutral(1), (40, 40))
    check_agent_pos(e, 0, (20, 20))
    check_agent_pos(e, 1, (40, 40))
    check_env(
        e,
        neutrals=2,
        attackers=0,
        defenders=0,
        people_of_interest=0,
        pois=0,
        walls=0,
    )


def test_autoid_neutral_creation():
    e = create_blank_environment()
    e.addNeutral(Neutral(0), (20, 20))
    e.addNeutral(Neutral(e.getNextId()), (40, 40))
    check_agent_pos(e, 0, (20, 20))
    check_agent_pos(e, 1, (40, 40))
    check_env(
        e,
        neutrals=2,
        attackers=0,
        defenders=0,
        people_of_interest=0,
        pois=0,
        walls=0,
    )


def test_inv_id_neutral_creation():
    e = create_blank_environment()
    e.addNeutral(Neutral(0), (20, 20))
    try:
        e.addNeutral(Neutral(0), (40, 40))
        raise NotImplementedError("An agent has been added with an existing id without raising an exception.")
    except ValueError:
        pass


def test_valid_personOfInterest_creation():
    e = create_blank_environment()
    e.addPersonOfInterest(PersonOfInterest(0), (10, 20))
    check_env(
        e,
        people_of_interest=1,
        attackers=0,
        defenders=0,
        neutrals=0,
        pois=0,
        walls=0,
    )
    check_agent_pos(e, 0, (10, 20))


def test_inv_pos_personOfInterest_creation():
    e = create_blank_environment()
    try:
        e.addPersonOfInterest(PersonOfInterest(0), (120, 20))
        raise NotImplementedError("The agent has been added outside the sim_map without raising an exception.")
    except ValueError:
        pass


def test_inv_class_personOfInterest_creation():
    e = create_blank_environment()
    try:
        e.addPersonOfInterest(Attacker(0), (12, 20))
        raise NotImplementedError("A Attacker has been added as PersonOfInterest without raising an exception.")
    except ValueError:
        pass


def test_two_personOfInterest_creation():
    e = create_blank_environment()
    e.addPersonOfInterest(PersonOfInterest(0), (20, 20))
    e.addPersonOfInterest(PersonOfInterest(1), (40, 40))
    check_agent_pos(e, 0, (20, 20))
    check_agent_pos(e, 1, (40, 40))
    check_env(
        e,
        people_of_interest=2,
        attackers=0,
        defenders=0,
        neutrals=0,
        pois=0,
        walls=0,
    )


def test_autoid_personOfInterest_creation():
    e = create_blank_environment()
    e.addPersonOfInterest(PersonOfInterest(0), (20, 20))
    e.addPersonOfInterest(PersonOfInterest(e.getNextId()), (40, 40))
    check_agent_pos(e, 0, (20, 20))
    check_agent_pos(e, 1, (40, 40))
    check_env(
        e,
        people_of_interest=2,
        attackers=0,
        defenders=0,
        neutrals=0,
        pois=0,
        walls=0,
    )


def test_inv_id_personOfInterest_creation():
    e = create_blank_environment()
    e.addPersonOfInterest(PersonOfInterest(0), (20, 20))
    try:
        e.addPersonOfInterest(PersonOfInterest(0), (40, 40))
        raise NotImplementedError("An agent has been added with an existing id without raising an exception.")
    except ValueError:
        pass


##################################################################
########################## Combinations ##########################
##################################################################

def test_autoid_defender_attacker_creation():
    e = create_blank_environment()
    e.addAttacker(Attacker(0), (20, 20))
    e.addDefender(Defender(e.getNextId()), (40, 40))
    check_agent_pos(e, 0, (20, 20))
    check_agent_pos(e, 1, (40, 40))
    check_env(
        e,
        defenders=1,
        attackers=1,
        people_of_interest=0,
        neutrals=0,
        pois=0,
        walls=0,
    )


def test_inv_pos_defender_attacker_creation():
    e = create_blank_environment()
    e.addAttacker(Attacker(0), (20, 20))
    try:
        e.addDefender(Defender(e.getNextId()), (20, 20))
        raise NotImplementedError("An agent has been added to an occupied location without rasising an exception.")
    except ValueError:
        pass


def test_inv_id_defender_attacker_creation():
    e = create_blank_environment()
    e.addAttacker(Attacker(0), (20, 20))
    try:
        e.addDefender(Defender(0), (40, 40))
        raise NotImplementedError("An agent has been added with an existing id without raising an exception.")
    except ValueError:
        pass

    check_agent_pos(e, 0, (20, 20))
    check_env(
        e,
        defenders=0,
        attackers=1,
        people_of_interest=0,
        neutrals=0,
        pois=0,
        walls=0,
    )


def test_autoid_3_agents_creation():
    e = create_blank_environment()
    e.addAttacker(Attacker(0), (20, 20))
    e.addDefender(Defender(e.getNextId()), (40, 40))
    e.addNeutral(Neutral(e.getNextId()), (50, 50))
    check_agent_pos(e, 0, (20, 20))
    check_agent_pos(e, 1, (40, 40))
    check_env(
        e,
        defenders=1,
        attackers=1,
        people_of_interest=0,
        neutrals=1,
        pois=0,
        walls=0,
    )


def test_autoid_4_agents_creation():
    e = create_blank_environment()
    e.addAttacker(Attacker(0), (20, 20))
    e.addDefender(Defender(e.getNextId()), (40, 40))
    e.addNeutral(Neutral(e.getNextId()), (30, 30))
    e.addPersonOfInterest(PersonOfInterest(e.getNextId()), (10, 10))
    check_agent_pos(e, 0, (20, 20))
    check_agent_pos(e, 1, (40, 40))
    check_agent_pos(e, 2, (30, 30))
    check_agent_pos(e, 3, (10, 10))
    check_env(
        e,
        defenders=1,
        attackers=1,
        people_of_interest=1,
        neutrals=1,
        pois=0,
        walls=0,
    )


##################################################################
########################## Random tests ##########################
##################################################################

"""
With x_size = 50 and y_size = 60, the probability of having two entities into the same slot is 1/(50*60) * 1/(50*60).
Having 6 classes of entities, in order to have about y cases of conflicting position over t test cases we should set the
    number of instances (n) for each entity to:
        (6*n over 2) * 1/(50*60)^2 = y/t.

To set y/t=0.2 => (sum_k=2^n(n!/(k!*(n-k)!))) * 1/(50*60)^2 = 0.2 => n = 21
"""

map_size_x = 50
map_size_y = 60
max_instances = 21
test_cases = 100


def random_values():
    return [(random.randint(0, map_size_x), random.randint(0, map_size_y)) for _ in
            range(0, random.randint(0, max_instances))]


testdata = [
    [random_values()] * 6 for i in range(test_cases)
]


@pytest.mark.parametrize("attackers, defenders, neutrals, peopleoi, pois, walls", testdata)
def test_scenario(attackers, defenders, neutrals, peopleoi, pois, walls):
    sim_map = np.zeros((map_size_y, map_size_x), dtype=np.int8)
    e = Environment(sim_map)

    for i, a in enumerate(attackers):
        try:
            e.addAttacker(Attacker(e.getNextId()), a)
            if a in attackers[:i]:
                raise NotImplementedError(
                    "An agent has been added to an occupied location without rasising an exception.")
        except ValueError:
            pass

    for i, a in enumerate(defenders):
        try:
            e.addDefender(Defender(e.getNextId()), a)
            if a in attackers or a in defenders[:i]:
                raise NotImplementedError(
                    "An agent has been added to an occupied location without rasising an exception.")
        except ValueError:
            pass

    for i, a in enumerate(neutrals):
        try:
            e.addNeutral(Neutral(e.getNextId()), a)
            if a in attackers or a in defenders or a in neutrals[:i]:
                raise NotImplementedError(
                    "An agent has been added to an occupied location without rasising an exception.")
        except ValueError:
            pass

    for i, a in enumerate(peopleoi):
        try:
            e.addPersonOfInterest(PersonOfInterest(e.getNextId()), a)
            if a in attackers or a in defenders or a in neutrals or a in peopleoi[:i]:
                raise NotImplementedError(
                    "An agent has been added to an occupied location without rasising an exception.")
        except ValueError:
            pass

    for i, a in enumerate(pois):
        try:
            e.addPOI(e.getNextId(), a)
            if a in attackers or a in defenders or a in neutrals or a in peopleoi or a in pois[:i]:
                raise NotImplementedError(
                    "An agent has been added to an occupied location without rasising an exception.")
        except ValueError:
            pass

    for i, a in enumerate(walls):
        try:
            e.addWall(e.getNextId(), a)
            if a in attackers or a in defenders or a in neutrals or a in peopleoi or a in pois or a in walls[:i]:
                raise NotImplementedError(
                    "An agent has been added to an occupied location without rasising an exception.")
        except ValueError:
            pass


##################################################################
################### Tests on the dynamic part ####################
##################################################################

def test_removal():
    e = create_blank_environment()
    a = Attacker(0)
    pos = (20, 20)
    e.addAttacker(a, pos)
    assert e.getAgent(a.id) == a, "The returned agent does not correspond to the added one."
    e.remove(0)
    assert len(e.getAgents()) == 0, "The agent has not been removed."
    try:
        e.getAgent(a.id)
        raise NotImplementedError("Tried to retrieve a removed id from the environment without raising an exception.")
    except ValueError:
        pass
    assert e.is_free(pos), "Even if the agent has been removed, the position is not free."


min_instances = 0
max_instances = 10
n_tests = 100

rint = partial(random.randint, min_instances, max_instances)

removal_test_data = [
    (rint(), rint(), rint(), rint(), rint(), rint()) for _ in range(n_tests)
]


def _gen_free_position(e):
    rnd = random.randint(0, 99), random.randint(0, 99)
    while not e.is_free(rnd):
        rnd = random.randint(0, 99), random.randint(0, 99)
    return rnd


@pytest.mark.parametrize("attackers, defenders, neutrals, peopleoi, pois, walls", removal_test_data)
def test_removal(attackers, defenders, neutrals, peopleoi, pois, walls):
    e = create_blank_environment()
    entities = [
        (attackers, Attacker),
        (defenders, Defender),
        (neutrals, Neutral),
        (peopleoi, PersonOfInterest),
        (pois, PointOfInterest),
        (walls, Wall)
    ]
    for entity_count, entity_class in entities:
        for a in range(entity_count):
            pos = _gen_free_position(e)
            e.add(entity_class(e.getNextId()), pos)
            assert not e.is_free(pos), "Even if the agent has been added, the position is still free."

    env_entities = e.getAllEntities()
    assert len(env_entities) == sum(
        [attackers, defenders, neutrals, peopleoi, pois, walls]), "Some entities haven't been added."

    random.shuffle(env_entities)

    if env_entities and len(env_entities) > 0:
        for ctr, ent in enumerate(env_entities):
            pos = e.getLocation(ent.id)
            e.remove(ent.id)
            assert e.is_free(pos), "Even if the agent has been removed, the position is not free."
            n_ent = len(e.getAllEntities())
            assert n_ent == len(env_entities) - (
                    ctr + 1), "Expected {} entities. Got {}.".format(n_ent, len(env_entities))


class DummyAgent(Attacker):
    def __init__(self, id, action):
        super(DummyAgent, self).__init__(id)
        self._action = action

    def feedInput(self, state):
        pass

    def getOutput(self):
        return self._action


rint = random.randint
m = 5
M = 10

testdata = [
    (rint(m, M), rint(m, M), DummyAgent(0, rint(0, 5)), (rint(1, M - 1), rint(1, M - 1))) for _ in
    range(n_tests)
]


def _correct(c, ax_size):
    n = c % ax_size
    if n == 0:
        n += 1
    elif n == ax_size - 1:
        n -= 1
    return n


@pytest.mark.parametrize("map_height, map_width, agent, agent_position", testdata)
def test_movement(map_height, map_width, agent, agent_position):
    # Correct agent position in case it's outside the map
    ag_pos = (_correct(agent_position[0], map_height), _correct(agent_position[1], map_width))

    sim_map = np.zeros((map_height, map_width), dtype=np.int8)
    e = Environment(sim_map)

    e.addAttacker(agent, ag_pos)

    # Build walls on the border
    for i in range(1, map_height - 1):
        e.addWall(e.getNextId(), (i, 0))
        e.addWall(e.getNextId(), (i, map_width - 1))
    for i in range(1, map_width - 1):
        e.addWall(e.getNextId(), (0, i))
        e.addWall(e.getNextId(), (map_height - 1, i))

    n_steps = None
    coord = None
    wallpos = None
    if agent._action == Agent.moveDown:
        coord = 0
        coef = 1
        wallpos = map_height - 1
        n_steps = wallpos - ag_pos[coord]
    elif agent._action == Agent.moveUp:
        coord = 0
        coef = -1
        wallpos = 0
        n_steps = ag_pos[coord]
    elif agent._action == Agent.moveRight:
        coord = 1
        coef = 1
        wallpos = map_width - 1
        n_steps = wallpos - ag_pos[coord]
    elif agent._action == Agent.moveLeft:
        coord = 1
        coef = -1
        wallpos = 0
        n_steps = ag_pos[coord]
    else:
        coord = -1
        n_steps = 10000

    for i in range(n_steps - 1):
        old_pos = e.getLocation(agent.id)
        if coord >= 0:
            e.moveAgentTo(agent.id, agent.getOutput())
        pos = e.getLocation(agent.id)
        if coord >= 0:
            assert old_pos != pos
            assert e.is_free(old_pos), "Even if the agent moved, the position is not free"
            assert abs(pos[coord] - ag_pos[coord]) == i + 1, \
                "Expected distance from starting point: {}. Actual: {}.".format(i, abs(pos[coord] - ag_pos[coord]))
            assert pos[abs(1 - coord)] == ag_pos[abs(1 - coord)], "The agent moved in an unexpected way."
        else:
            assert pos == ag_pos, "The agent moved but it shouldn't have moved."

    if coord > 0:
        e.moveAgentTo(agent.id, agent.getOutput())
        assert e.getLocation(agent.id)[coord] != wallpos, "The agent hit a wall."


class DummyDefender(Defender):
    def __init__(self, id):
        super().__init__(id)
        self._state = None

    def feedInput(self, state):
        s = len(state)
        c = int(round(s / 2))
        if np.sum(state[c - 1:c + 1, c - 1:c + 1]) > 0:
            self._state = Agent.capture
        self._state = Agent.stay

    def getOutput(self):
        return self._state


class DummyAttacker(DummyAgent, Attacker):
    pass


@pytest.mark.parametrize("map_height, map_width, agent, agent_position", testdata)
def test_capture(map_height, map_width, agent, agent_position, ):
    # Correct agent position in case it's outside the map
    ag_pos = (_correct(agent_position[0], map_height), _correct(agent_position[1], map_width))

    sim_map = np.zeros((map_height, map_width), dtype=np.int8)
    e = Environment(sim_map)

    e.addAttacker(agent, ag_pos)

    # Build walls on the border
    for i in range(1, map_height - 1):
        e.addWall(e.getNextId(), (i, 0))
        e.addWall(e.getNextId(), (i, map_width - 1))
    for i in range(1, map_width - 1):
        e.addWall(e.getNextId(), (0, i))
        e.addWall(e.getNextId(), (map_height - 1, i))

    n_steps = None
    coord = None
    if agent._action == Agent.moveDown:
        coord = 0
        n_steps = map_height - ag_pos[coord]
        target_pos = (map_height - 2, ag_pos[1])
        if target_pos == ag_pos:
            return
        e.addDefender(DummyDefender(e.getNextId()), target_pos)
    elif agent._action == Agent.moveUp:
        coord = 0
        n_steps = ag_pos[coord]
        target_pos = (1, ag_pos[1])
        if target_pos == ag_pos:
            return
        e.addDefender(DummyDefender(e.getNextId()), target_pos)
    elif agent._action == Agent.moveRight:
        coord = 1
        n_steps = map_width - ag_pos[coord]
        target_pos = (ag_pos[0], map_width - 2)
        if target_pos == ag_pos:
            return
        e.addDefender(DummyDefender(e.getNextId()), target_pos)
    elif agent._action == Agent.moveLeft:
        coord = 1
        n_steps = ag_pos[coord]
        target_pos = (ag_pos[0], 1)
        if target_pos == ag_pos:
            return
        e.addDefender(DummyDefender(e.getNextId()), target_pos)
    else:
        coord = -1
        n_steps = 100

    for i in range(n_steps - 1):
        e.executeTimestep()
        try:
            e.getLocation(agent.id)
        except:
            print()

    try:
        if coord > 0:
            e.executeTimestep()
            e.getLocation(agent.id)
            raise NotImplementedError("The agent should have been captured. \nAgents locations: {}".format(
                [a.position for a in e._entities.values() if not isinstance(a.entity, Wall)]))
    except KeyError:
        pass


def test_conflicting_move():
    sim_map = np.zeros((10, 10), dtype=np.int8)
    e = Environment(sim_map)

    e.addAttacker(DummyAttacker(0, Agent.moveRight), (1, 1))
    e.addAttacker(DummyAttacker(1, Agent.moveLeft), (1, 3))

    e.executeTimestep()
    assert e.getLocation(0) == (1, 1), "Agent 0 moved, but instead it should have been in the same position"
    assert e.getLocation(1) == (1, 3), "Agent 0 moved, but instead it should have been in the same position"


def test_max_timesteps():
    sim_map = np.zeros((10, 10), dtype=np.int8)
    max_timesteps = 50
    e = Environment(sim_map, max_timesteps=max_timesteps)
    e.addAttacker(DummyAgent(0, Agent.stay), (2, 2))
    for k in range(max_timesteps):
        e.executeTimestep()
        assert not e.game_ended(), "The game finished earlier, at step {}".format(k)
    e.executeTimestep()
    print(e.timestep, e.res)
    assert e.game_ended(), "The game did not finish"


class RepresentationTestAgent(Attacker):
    def __init__(self, id):
        super().__init__(id)
        self.observation = None

    def feedInput(self, state):
        self.observation = state

    def getOutput(self):
        return self.stay


repr_test_data0 = np.random.randint(2, 8, (20, 2))


@pytest.mark.parametrize("agposy, agposx", repr_test_data0)
def test_representation_0(agposy, agposx):
    los = 2
    w = h = 10
    sim_map = np.zeros((h, w), dtype=np.int8)
    for a in range(-2, 3):
        for b in range(-2, 3):
            if a != 0 or b != 0:
                e = Environment(sim_map, line_of_sight=los)
                agent = RepresentationTestAgent(0)
                e.addAttacker(agent, (agposy, agposx))
                e.addWall(1, (agposy - a, agposx - b))
                e.executeTimestep()

                los_r = min(los, w - agposx - 1)
                los_d = min(los, h - agposy - 1)
                los_l = min(los, agposx)
                los_u = min(los, agposy)

                for i in range(2 * los + 1):
                    for j in range(2 * los + 1):
                        if not (i == los_u - a and j == los_l - b):
                            assert agent.observation[
                                       i, j] != Map.wall, "A wall has been found in position {},{}".format(i, j)
                        else:
                            assert agent.observation[
                                       i, j] == Map.wall, "Wall not found in position {},{}.\n{}".format(i, j, agent.observation)
                assert agent.observation[los_u, los_l] == Map.attacker, "The agent didn't see itself"


@pytest.mark.parametrize("agposy, agposx", repr_test_data0)
def test_representation_1(agposy, agposx):
    los = 3
    w = h = 10
    sim_map = np.zeros((h, w), dtype=np.int8)
    for a in range(-1, 2):
        for b in range(-1, 2):
            if not (a == 0 and b == 0):
                e = Environment(sim_map, line_of_sight=los)
                agent = RepresentationTestAgent(0)
                e.addWall(1, (agposy + a, agposx + b))
                e.addAttacker(agent, (agposy, agposx))
                e.executeTimestep()

                los_r = min(los, w - agposx - 1)
                los_d = min(los, h - agposy - 1)
                los_l = min(los, agposx)
                los_u = min(los, agposy)
                for i in range(los_u + los_d + 1):
                    for j in range(los_l + los_r + 1):
                        if not (i == los_u + a and j == los_l + b):  # a and b are shifted
                            assert agent.observation[
                                       i, j] != Map.wall, "A wall has been found in position {},{}.".format(i, j)
                        else:
                            assert agent.observation[i, j] == Map.wall, "Wall not found at {},{}.\n{}".format(i, j, agent.observation)
                assert agent.observation[los_u, los_l] == Map.attacker, "The agent didn't see itself"


repr_test_data1 = [2, 3, 4]


@pytest.mark.parametrize("los", repr_test_data1)
def test_border_representation(los):
    agent = RepresentationTestAgent(0)
    sim_map = np.zeros((10, 10), dtype=np.int8)
    for y in [0, 9]:
        for x in [0, 9]:
            if y == 0:
                step_a = 1
            else:
                step_a = -1
            if x == 0:
                step_b = 1
            else:
                step_b = -1
            for a in range(los):
                for b in range(los):
                    if not (a == 0 and b == 0):
                        e = Environment(sim_map, line_of_sight=los)
                        e.addAttacker(agent, (y + step_a * a, x + step_b * b))
                        e.addWall(e.getNextId(), (0, 0))
                        e.addWall(e.getNextId(), (0, 9))
                        e.addWall(e.getNextId(), (9, 0))
                        e.addWall(e.getNextId(), (9, 9))
                        e.executeTimestep()

                        obs = agent.observation
                        shape = obs.shape
                        tgty = 2 * los - (los - a) + 1
                        tgtx = 2 * los - (los - b) + 1
                        assert shape[0] == tgty, "Height of the observation should be {} but it's {}. a is {}".format(
                            tgty, shape[0], a)
                        assert shape[1] == tgtx, "Width of the observation should be {} but it's {}. b is {}".format(
                            tgtx, shape[1], b)

                        if y < 5:
                            if x < 5:
                                assert obs[0, 0] == Map.wall
                            else:
                                assert obs[0, -1] == Map.wall
                        else:
                            if x < 5:
                                assert obs[-1, 0] == Map.wall
                            else:
                                assert obs[-1, -1] == Map.wall


if __name__ == '__main__':
    test_representation_0(*repr_test_data0[0])
