import sys
import numpy as np

sys.path.append(".")
sys.path.append("..")

from simulator_map import Map
from entities import Void, Wall, PointOfInterest, Agent, Attacker, Defender, Neutral, PersonOfInterest


def check_size(m, height, width):
    assert m.height == height, "Incorrect height"
    assert m.width == width, "Incorrect width"


def var_name(**a):
    return list(a.keys())[0]


def get_var_name(var):
    return var_name(var=var)


def check_var(var, expected_value, name):
    """
    Checks the value of a var if its expected value is >= 0
    :param var: The var to check
    :param expected_value: expected value
    """
    if expected_value >= 0:
        assert var == expected_value, "Expecting {} {}, got {}.".format(expected_value, name, var)


def _map_checks(m, width, height, expected_agents, expected_attackers, expected_defenders, expected_neutral,
                expected_people_of_interest, expected_point_of_interest, expected_walls):
    check_size(m, height, width)

    agents = 0
    attackers = 0
    defenders = 0
    neutral = 0
    people_of_interest = 0
    points_of_interest = 0
    walls = 0

    for i in range(20):
        for j in range(20):
            element = m[j, i]
            if element == m.attacker or element == m.defender or element == m.neutral or element == m.person_of_interest:
                agents += 1
                if element == m.attacker:
                    attackers += 1
                elif element == m.defender:
                    defenders += 1
                elif element == m.neutral:
                    neutral += 1
                elif element == m.person_of_interest:
                    people_of_interest += 1
            elif element == m.point_of_interest:
                points_of_interest += 1
            elif element == m.wall:
                walls += 1

    check_var(agents, expected_agents, 'agents')
    check_var(attackers, expected_attackers, 'attackers')
    check_var(defenders, expected_defenders, 'defenders')
    check_var(neutral, expected_neutral, 'neutrals')
    check_var(people_of_interest, expected_people_of_interest, 'people of interest')
    check_var(points_of_interest, expected_point_of_interest, 'points of interest')
    check_var(walls, expected_walls, 'walls')

    return m


def map_checks(filepath, width, height, expected_agents, expected_attackers, expected_defenders, expected_neutral,
               expected_people_of_interest, expected_point_of_interest, expected_walls):
    m = Map(filepath=filepath)
    _map_checks(m, width, height, expected_agents, expected_attackers, expected_defenders, expected_neutral,
                expected_people_of_interest, expected_point_of_interest, expected_walls)
    return m


def test_empty_map_loading():
    filepath = "tests/files/emptymap_20x20.map"
    map_checks(
        filepath=filepath,
        width=20,
        height=20,
        expected_agents=0,
        expected_attackers=0,
        expected_defenders=0,
        expected_neutral=0,
        expected_people_of_interest=0,
        expected_point_of_interest=0,
        expected_walls=0
    )


def test_h_rect_empty_map_loading():
    filepath = "tests/files/emptymap_30x20.map"
    map_checks(
        filepath=filepath,
        width=20,
        height=30,
        expected_agents=0,
        expected_attackers=0,
        expected_defenders=0,
        expected_neutral=0,
        expected_people_of_interest=0,
        expected_point_of_interest=0,
        expected_walls=0
    )


def test_v_rect_empty_map_loading():
    filepath = "tests/files/emptymap_20x30.map"
    map_checks(
        filepath=filepath,
        width=30,
        height=20,
        expected_agents=0,
        expected_attackers=0,
        expected_defenders=0,
        expected_neutral=0,
        expected_people_of_interest=0,
        expected_point_of_interest=0,
        expected_walls=0
    )


def test_one_agent_map_loading():
    filepath = "tests/files/one_agent_map_20x20.map"
    map_checks(
        filepath=filepath,
        width=20,
        height=20,
        expected_agents=1,
        expected_attackers=-1,
        expected_defenders=-1,
        expected_neutral=0,
        expected_people_of_interest=0,
        expected_point_of_interest=0,
        expected_walls=0
    )


def test_one_agent_with_poi_map_loading():
    filepath = "tests/files/one_agent_poi_map_20x20.map"
    map_checks(
        filepath=filepath,
        width=20,
        height=20,
        expected_agents=1,
        expected_attackers=-1,
        expected_defenders=-1,
        expected_neutral=0,
        expected_people_of_interest=0,
        expected_point_of_interest=1,
        expected_walls=0
    )


def test_two_opposite_agents_map_loading():
    filepath = "tests/files/two_opposite_agents_map_20x20.map"
    map_checks(
        filepath=filepath,
        width=20,
        height=20,
        expected_agents=2,
        expected_attackers=1,
        expected_defenders=1,
        expected_neutral=0,
        expected_people_of_interest=0,
        expected_point_of_interest=0,
        expected_walls=0
    )


def test_8_att_2_def_map_loading():
    filepath = "tests/files/8_att_2_def_map_20x20.map"
    map_checks(
        filepath=filepath,
        width=20,
        height=20,
        expected_agents=10,
        expected_attackers=8,
        expected_defenders=2,
        expected_neutral=0,
        expected_people_of_interest=0,
        expected_point_of_interest=0,
        expected_walls=0
    )


def test_1_neutral_map_loading():
    filepath = "tests/files/1_neutral_map_20x20.map"
    map_checks(
        filepath=filepath,
        width=20,
        height=20,
        expected_agents=11,
        expected_attackers=8,
        expected_defenders=2,
        expected_neutral=1,
        expected_people_of_interest=0,
        expected_point_of_interest=0,
        expected_walls=0
    )


def test_10_neutral_map_loading():
    filepath = "tests/files/10_neutral_map_20x20.map"
    map_checks(
        filepath=filepath,
        width=20,
        height=20,
        expected_agents=20,
        expected_attackers=8,
        expected_defenders=2,
        expected_neutral=10,
        expected_people_of_interest=0,
        expected_point_of_interest=0,
        expected_walls=0
    )


def test_multiple_agents_1_pploi_map_loading():
    filepath = "tests/files/1_pploi_map_20x20.map"
    map_checks(
        filepath=filepath,
        width=20,
        height=20,
        expected_agents=21,
        expected_attackers=8,
        expected_defenders=2,
        expected_neutral=10,
        expected_people_of_interest=1,
        expected_point_of_interest=0,
        expected_walls=0
    )


def test_multiple_agents_10_pploi_map_loading():
    filepath = "tests/files/10_pploi_map_20x20.map"
    map_checks(
        filepath=filepath,
        width=20,
        height=20,
        expected_agents=30,
        expected_attackers=8,
        expected_defenders=2,
        expected_neutral=10,
        expected_people_of_interest=10,
        expected_point_of_interest=0,
        expected_walls=0
    )


def test_multiple_agents_1_poi_map_loading():
    filepath = "tests/files/1_poi_map_20x20.map"
    map_checks(
        filepath=filepath,
        width=20,
        height=20,
        expected_agents=20,
        expected_attackers=8,
        expected_defenders=2,
        expected_neutral=10,
        expected_people_of_interest=0,
        expected_point_of_interest=1,
        expected_walls=0
    )


def test_multiple_agents_10_poi_map_loading():
    filepath = "tests/files/10_poi_map_20x20.map"
    map_checks(
        filepath=filepath,
        width=20,
        height=20,
        expected_agents=20,
        expected_attackers=8,
        expected_defenders=2,
        expected_neutral=10,
        expected_people_of_interest=0,
        expected_point_of_interest=10,
        expected_walls=0
    )


def test_multiple_agents_10_poi_10_pploi_map_loading():
    filepath = "tests/files/10_poi_10_pploi_map_20x20.map"
    map_checks(
        filepath=filepath,
        width=20,
        height=20,
        expected_agents=30,
        expected_attackers=8,
        expected_defenders=2,
        expected_neutral=10,
        expected_people_of_interest=10,
        expected_point_of_interest=10,
        expected_walls=0
    )


def test_multiple_agents_10_poi_10_pploi_10_wls_map_loading():
    filepath = "tests/files/10_wls_map_20x20.map"
    m = map_checks(
        filepath=filepath,
        width=20,
        height=20,
        expected_agents=30,
        expected_attackers=8,
        expected_defenders=2,
        expected_neutral=10,
        expected_people_of_interest=10,
        expected_point_of_interest=10,
        expected_walls=10
    )
    assert sum([m[0, i] == m.wall for i in range(10)]), "Map is not translated properly"


def test_matrix_loading():
    filepath = "tests/files/10_wls_map_20x20.map"
    m = np.loadtxt(filepath, dtype=np.int8)
    m = Map(matrix=m)

    _map_checks(
        m=m,
        width=20,
        height=20,
        expected_agents=30,
        expected_attackers=8,
        expected_defenders=2,
        expected_neutral=10,
        expected_people_of_interest=10,
        expected_point_of_interest=10,
        expected_walls=10
    )


def test_no_attribute():
    filepath = "tests/files/10_wls_map_20x20.map"
    m = np.loadtxt(filepath, dtype=np.int8)
    try:
        m = Map()
        raise NotImplementedError("Constructor without arguments succeeded")
    except AttributeError:
        pass


def test_both_attributes():
    filepath = "tests/files/10_wls_map_20x20.map"
    m = np.loadtxt(filepath, dtype=np.int8)
    try:
        m = Map(matrix=m, filepath=filepath)
        raise NotImplementedError("Constructor with both matrix arguments succeeded")
    except AttributeError:
        pass


def test_bad_formed_matrix_empty():
    matrix = [""] * 10
    try:
        Map(matrix=matrix)
        raise NotImplementedError("Empty matrix passed the test")
    except ValueError:
        pass


def test_bad_formed_matrix_badshape():
    matrix = [['0'] * 10] * 10
    matrix[3] = matrix[3][:-1]
    try:
        Map(matrix=matrix)
        raise NotImplementedError("Matrix with bad shape passed the test")
    except ValueError:
        pass


def test_bad_formed_matrix_badchar_0():
    matrix = [['0'] * 10] * 10
    matrix[3][3] = 'k'

    try:
        Map(matrix=matrix)
        raise AssertionError("Matrix with bad_chars passed the test")
    except ValueError:
        pass


def test_bad_formed_matrix_badchar_1():
    matrix = [['0'] * 10] * 10
    matrix[3][3] = 'A'

    try:
        Map(matrix=matrix)
        raise AssertionError("Matrix with bad_chars passed the test")
    except ValueError:
        pass


def test_bad_formed_matrix_badchar_2():
    matrix = [['0'] * 10] * 10
    matrix[3][3] = 'D'

    try:
        Map(matrix=matrix)
        raise NotImplementedError("Matrix with bad_chars passed the test")
    except ValueError:
        pass
