import numpy as np
import neat
import os
import multiprocessing
import random
import pickle
from neat.reporting import BaseReporter

from environment import Environment
from entities import Attacker, Agent

name = "exp3"
MAP_DIM = 5

RUN = 10

def manhattan_distance(a,b):
    return abs(b[0]-a[0]) + abs(b[1]-a[1])

def generate_positions():
    end_positions = [(random.randint(0, MAP_DIM - 1), random.randint(0, MAP_DIM - 1)) for i in range(RUN)]
    start_positions = []
    for _ in range(RUN):
        x = (random.randint(0, MAP_DIM - 1), random.randint(0, MAP_DIM - 1))
        while x in end_positions:
            x = (random.randint(0, MAP_DIM - 1), random.randint(0, MAP_DIM - 1))
        start_positions.append(x)
    return start_positions, end_positions

def choose_best_positions():
    import scipy.spatial.distance
    N_ROUNDS = 100
    best = None
    best_value = 0
    for _ in range(N_ROUNDS):
        d = 0
        possible_start, possible_end = generate_positions()
        vectors = []
        for i in range(len(possible_start)):
            a = possible_start[i]
            b = possible_end[i]
            c = (b[0] - a[0], b[1] - a[1])
            vectors.append(list(c))
        for i in range(len(vectors)):
            for j in range(len(vectors)):
                d += scipy.spatial.distance.cosine(vectors[i], vectors[j])
        if d > best_value:
            best = (possible_start, possible_end)
            best_value = d       
    return best[0], best[1]

start_positions, end_positions = choose_best_positions()

class BestGenomeReporter(BaseReporter):
    """Uses `print` to output information about the run; an example reporter class."""
    def __init__(self, name):
        #self.show_species_detail = show_species_detail
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0
        self.store_mean = []
        self.store_best = []
        self.name = name

    def post_evaluate(self, config, population, species, best_genome):
        fitnesses = [c.fitness for c in population.values()]
        fit_mean = np.mean(fitnesses)
        self.store_mean.append(fit_mean)
        self.store_best.append(best_genome.fitness)

    def found_solution(self, config, generation, best):
        f = open('{}.mean'.format(self.name), 'a+')
        f.write(' '.join(list(map(str, self.store_mean))) + '\n')
        f.close()
        f = open('{}.best'.format(self.name), 'a+')
        f.write(' '.join(list(map(str, self.store_best))) + '\n')
        f.close()

class NeuralAttacker(Attacker):
    def __init__(self, id, genome, config):
        super().__init__(id)
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.action = []

    def _find(self, state, id):
        for (x,y), value in np.ndenumerate(state):
            if value == id:
                # normalizzazione?
                return (x, y)
        return (-1, -1)

    def feedInput(self, state):
        def norm(a):
            if a > 0:
                return 1
            elif a < 0:
                return 2
            else:
                return 0        
        att_x, att_y = self._find(state, 1)
        poi_x, poi_y = self._find(state, 4)
        poi_x -= att_x
        poi_y -= att_y
        poi_x = norm(poi_x)
        poi_y = norm(poi_y)
        #if poi_x == 1 and poi_y == 0:
        #    poi_x = 0
        #elif poi_x == -1 and poi_y == 0:
        #    poi_x = 0
        self.action = self.net.activate([poi_x, poi_y])

    def getOutput(self):
        return int(np.argmax(self.action))

f = open("positions", 'w')
for i in range(len(end_positions)):
    f.write("({},{}) - ({},{})\n".format(start_positions[i][1], start_positions[i][0], end_positions[i][1], end_positions[i][0]))
f.close()


def eval_genome(genome, config):
    fitnesses = []
    ID_ATTACKER = 5
    attacker = NeuralAttacker(ID_ATTACKER, genome, config)

    for i in range(len(end_positions)):
        sim_map = np.zeros((MAP_DIM, MAP_DIM), dtype=np.int8)
        e = Environment(sim_map, show=False, max_timesteps=MAP_DIM*3, line_of_sight=MAP_DIM)

        y, x = end_positions[i]
        y1, x1 = start_positions[i]

        e.addPOI(3, (y, x))
        e.addAttacker(attacker, (y1, x1))

        steps = 0
        while not e.game_ended():
            steps+=1
            e.executeTimestep()
            
        fitness = manhattan_distance((x,y), (x1, y1)) - steps
        fitnesses.append(fitness)

    # The genome's fitness is its worst performance across all runs.
    print(fitnesses)
    return np.mean(fitnesses)

# Final animation
def viz_winner_net(winner, config):
    runs_per_net = 100
    attacker = NeuralAttacker(5, winner, config)
    ok = 0
    for _ in range(runs_per_net):
        sim_map = np.zeros((MAP_DIM, MAP_DIM), dtype=np.int8)
        e = Environment(sim_map, show=False, max_timesteps=MAP_DIM * 3, line_of_sight=MAP_DIM)
        x = random.randint(0, MAP_DIM - 1)
        y = random.randint(0, MAP_DIM - 1)
        e.addPOI(3, (x, y))

        x1 = random.randint(0, MAP_DIM - 1)
        y1 = random.randint(0, MAP_DIM - 1)
        while x == x1 and y == y1:
            x1 = random.randint(0, MAP_DIM - 1)
            y1 = random.randint(0, MAP_DIM - 1)

        e.addAttacker(attacker, (x1, y1))

        while not e.game_ended():
            e.executeTimestep()

        if e.res == 0:
            ok += 1

    # The genome's fitness is its worst performance across all runs.
    print("OK: {} out of {}".format(ok, runs_per_net))
    f = open('{}.test5'.format(name), 'a+')
    f.write('{} out of 100\n'.format(ok))
    f.close()

# Final animation
def viz_winner_net1(winner, config):
    MAP_DIM = 10
    runs_per_net = 100
    attacker = NeuralAttacker(5, winner, config)
    ok = 0
    for _ in range(runs_per_net):
        sim_map = np.zeros((MAP_DIM, MAP_DIM), dtype=np.int8)
        e = Environment(sim_map, show=True, max_timesteps=MAP_DIM * 3, line_of_sight=MAP_DIM)
        x = random.randint(0, MAP_DIM - 1)
        y = random.randint(0, MAP_DIM - 1)
        e.addPOI(3, (x, y))

        x1 = random.randint(0, MAP_DIM - 1)
        y1 = random.randint(0, MAP_DIM - 1)
        while x == x1 and y == y1:
            x1 = random.randint(0, MAP_DIM - 1)
            y1 = random.randint(0, MAP_DIM - 1)

        e.addAttacker(attacker, (x1, y1))

        while not e.game_ended():
            e.executeTimestep()

        if e.res == 0:
            ok += 1

    # The genome's fitness is its worst performance across all runs.
    print("OK: {} out of {}".format(ok, runs_per_net))
    f = open('{}.test10'.format(name), 'a+')
    f.write('{} out of 100\n'.format(ok))
    f.close()

# Final animation
def viz_winner_net2(winner, config):
    MAP_DIM = 20
    runs_per_net = 100
    attacker = NeuralAttacker(5, winner, config)
    ok = 0
    for _ in range(runs_per_net):
        sim_map = np.zeros((MAP_DIM, MAP_DIM), dtype=np.int8)
        e = Environment(sim_map, show=False, max_timesteps=MAP_DIM * 3, line_of_sight=MAP_DIM)
        x = random.randint(0, MAP_DIM - 1)
        y = random.randint(0, MAP_DIM - 1)
        e.addPOI(3, (x, y))

        x1 = random.randint(0, MAP_DIM - 1)
        y1 = random.randint(0, MAP_DIM - 1)
        while x == x1 and y == y1:
            x1 = random.randint(0, MAP_DIM - 1)
            y1 = random.randint(0, MAP_DIM - 1)

        e.addAttacker(attacker, (x1, y1))

        while not e.game_ended():
            e.executeTimestep()

        if e.res == 0:
            ok += 1
    # The genome's fitness is its worst performance across all runs.
    print("OK: {} out of {}".format(ok, runs_per_net))
    f = open('{}.test20'.format(name), 'a+')
    f.write('{} out of 100\n'.format(ok))
    f.close()


# Loads NEAT configuration file
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'neat.config')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    config_path)


population = neat.Population(config)

# Adds statistics report
stats = neat.StatisticsReporter()
population.add_reporter(stats)
population.add_reporter(neat.StdOutReporter(True))
#population.add_reporter(neat.Checkpointer(10))
population.add_reporter(BestGenomeReporter(name))

# Parallel execution over all the available processors
pe = neat.ThreadedEvaluator(multiprocessing.cpu_count(), eval_genome)


# Runs the NEAT algorithm and returns the best network
winner = population.run(pe.evaluate)


viz_winner_net(winner, config)
viz_winner_net1(winner, config)
viz_winner_net2(winner, config)
# Saves winnder net
pickle_out = open("winner_exp3.pickle","wb")
pickle.dump(winner, pickle_out)
pickle_out.close()


#winner = pickle.load(open("winner_exp3.pickle","rb"))
#viz_winner_net1(winner, config)