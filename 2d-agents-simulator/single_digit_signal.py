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
        # navigation
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'neat.config')
        config_nav = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_path)
        nav_net = pickle.load(open("winner_exp3.pickle","rb"))
        self.net = neat.nn.FeedForwardNetwork.create(nav_net, config_nav)

        # communication
        self.net_rec = neat.ctrnn.CTRNN.create(genome, config, 1)
        self.action = []

    def _find(self, state, id):
        for (x,y), value in np.ndenumerate(state):
            if value == id:
                # normalizzazione?
                return (x, y)
        return (-1, -1)

    def listen(self, inputs, dt):
        self.out = self.net_rec.advance(inputs, dt, dt)
    
    def learn(self):
        self.poi_y = int(self.out[0]*MAP_DIM)
        #self.poi_x = int(self.out[1]*MAP_DIM)
        #print("PRED: {} {}".format(self.poi_y, self.poi_x))

    def feedInput(self, state):
        def norm(a):
            if a > 0:
                return 1
            elif a < 0:
                return -1
            else:
                return 0        
        att_x, att_y = self._find(state, 1)
        poi_y = self.poi_y
        poi_x = self.poi_x
        poi_x -= att_x
        poi_y -= att_y
        poi_x = norm(poi_x)
        poi_y = norm(poi_y)
        self.action = self.net.activate([poi_x, poi_y])

    def getOutput(self):
        return int(np.argmax(self.action))

f = open("positions", 'w')
for i in range(len(end_positions)):
    f.write("({},{}) - ({},{})\n".format(start_positions[i][1], start_positions[i][0], end_positions[i][1], end_positions[i][0]))
f.close()

class CompleteExtinctionException(Exception):
    pass

def eval_genomes(genomes, send_config, rec_config):
    fitnesses = []
    for genome in genomes:
        send = genome[0][1]
        rec = genome[1][1]
        net_send = neat.ctrnn.CTRNN.create(send, send_config, 1)
        ID_ATTACKER = 5
        attacker = NeuralAttacker(ID_ATTACKER, rec, rec_config)

        fitnesses = []
        for i in range(MAP_DIM):
            #sim_map = np.zeros((MAP_DIM, MAP_DIM), dtype=np.int8)
            #e = Environment(sim_map, show=False, max_timesteps=MAP_DIM*3, line_of_sight=MAP_DIM)

            y, x = end_positions[i]

            y = i

            #print("TRUE: {} {}".format(y, x))
            dt = 1
            t = 0
            T_MAX = 5
            norm_y, norm_x = y / MAP_DIM, x / MAP_DIM
            attacker.net_rec.reset()
            net_send.reset()
            while t < T_MAX:
                t += dt
                out1 = net_send.advance([norm_y], dt, dt)
                attacker.listen(out1, dt)
            attacker.learn()    
            fitness = 0
            #if x == attacker.poi_x:
            #    fitness += 10
            if y == attacker.poi_y:
                fitness += 10
            #if fitness == 20:
            #    fitness = 100
            fitnesses.append(fitness)
        print(fitnesses)

        if send.fitness is None:
            send.fitness = np.mean(fitnesses)
        else:
            send.fitness = max(np.mean(fitnesses), send.fitness)
        if rec.fitness is None:
            rec.fitness = np.mean(fitnesses)
        else:
            rec.fitness = max(np.mean(fitnesses), rec.fitness)

def run(population_send, population_rec, fitness_function=eval_genomes, n=None):
    """
    population_send is the population of the senders
    population_rec is the population of the receivers

    Runs NEAT's genetic algorithm for at most n generations.
    If n is None, run until solution is found or extinction occurs.
    """

    if (population_send.config.no_fitness_termination or population_send.config.no_fitness_termination) and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

    k = 0
    while n is None or k < n:
        k += 1

        population_send.reporters.start_generation(population_send.generation)
        population_rec.reporters.start_generation(population_send.generation)

        genomes_sender = list(population_send.population.items())
        n1 = len(genomes_sender)
        genomes_receiver = list(population_rec.population.items())
        n2 = len(genomes_receiver)

        import random
        genomes = []
        for i in range(n1):
            for j in range(n2):
                genomes.append((genomes_sender[i], genomes_receiver[j]))
        #for _ in range(400):
        #    genomes.append((genomes_sender[random.randint(0, n1 - 1)], genomes_receiver[random.randint(0, n2 - 1)]))

        # Evaluate all genomes using the user-provided function.
        fitness_function(genomes, population_send.config, population_rec.config)

        # Gather and report statistics.
        best_send = None
        best_rec = None
        for g in population_send.population.values():
            if g.fitness is None:
                g.fitness = -100
                #raise RuntimeError("Fitness not assigned to genome {}".format(g.key))

            if best_send is None or g.fitness > best_send.fitness:
                best_send = g

        for g in population_rec.population.values():
            if g.fitness is None:
                g.fitness = -100
            if best_rec is None or g.fitness > best_rec.fitness:
                best_rec = g

        population_send.reporters.post_evaluate(population_send.config, population_send.population, population_send.species, best_send)
        population_rec.reporters.post_evaluate(population_rec.config, population_rec.population, population_rec.species, best_rec)


        # Track the best genome ever seen.
        if population_send.best_genome is None or best_send.fitness > population_send.best_genome.fitness:
            population_send.best_genome = best_send
        
        if population_rec.best_genome is None or best_rec.fitness > population_rec.best_genome.fitness:
            population_rec.best_genome = best_rec

        # this is true for both since the fitness is shared
        if not population_send.config.no_fitness_termination:
            # End if the fitness threshold is reached.
            fv = population_send.fitness_criterion(g.fitness for g in population_send.population.values())
            if fv >= population_send.config.fitness_threshold:
                population_send.reporters.found_solution(population_send.config, population_send.generation, best_send)
                population_rec.reporters.found_solution(population_rec.config, population_rec.generation, best_rec)
                break

        # Create the next generation from the current generation.
        population_send.population = population_send.reproduction.reproduce(population_send.config, population_send.species,
                                                        population_send.config.pop_size, population_send.generation)
        population_rec.population = population_rec.reproduction.reproduce(population_rec.config, population_rec.species,
                                                        population_rec.config.pop_size, population_rec.generation)

        # Check for complete extinction.
        if not population_send.species.species or not population_rec.species.species:
            population_send.reporters.complete_extinction()
            population_rec.reporters.complete_extinction()

            # If requested by the user, create a completely new population,
            # otherwise raise an exception.
            if population_send.config.reset_on_extinction and population_rec.config.reset_on_extinction:
                population_send.population = population_send.reproduction.create_new(population_send.config.genome_type,
                                                                population_send.config.genome_config,
                                                                population_send.config.pop_size)
                population_rec.population = population_rec.reproduction.create_new(population_rec.config.genome_type,
                                                                population_rec.config.genome_config,
                                                                population_rec.config.pop_size)
            else:
                raise CompleteExtinctionException()

        # Divide the new population into species.
        population_send.species.speciate(population_send.config, population_send.population, population_send.generation)
        population_rec.species.speciate(population_rec.config, population_rec.population, population_rec.generation)

        population_send.reporters.end_generation(population_send.config, population_send.population, population_send.species)
        population_rec.reporters.end_generation(population_rec.config, population_rec.population, population_rec.species)

        population_send.generation += 1
        population_rec.generation += 1

    if population_send.config.no_fitness_termination and population_rec.config.no_fitness_termination:
        population_send.reporters.found_solution(population_send.config, population_send.generation, population_send.best_genome)
        population_send.reporters.found_solution(population_rec.config, population_rec.generation, population_rec.best_genome)

    return population_send.best_genome, population_rec.best_genome

def viz(send_genome, rec_genome, send_config, rec_config):
    net_send = neat.ctrnn.CTRNN.create(send_genome, send_config, 1)
    ID_ATTACKER = 5
    attacker = NeuralAttacker(ID_ATTACKER, rec_genome, rec_config)
    pos = []
    #for i in range(0, 5):
    #    for j in range(0,5):
    #        pos.append((i,j))
    cont = 0
    for i in range(5):
        pos.append((i,i))
    for y,x in pos:
        #y, x = random.randint(0, MAP_DIM + 1), random.randint(0, MAP_DIM + 1)
        #print("TRUE: {} {}".format(y, x))
        dt = 1
        t = 0
        T_MAX = 5
        norm_y, norm_x = y / MAP_DIM, x / MAP_DIM
        attacker.net_rec.reset()
        net_send.reset()
        sign = []
        while t < T_MAX:
            t += dt
            out1 = net_send.advance([norm_y], dt, dt)
            sign.append(out1)
            attacker.listen(out1, dt)
        attacker.learn()  
        print(sign)
        #print("PRED: {} {}".format(attacker.poi_y, attacker.poi_y))
        if attacker.poi_y != y: #and attacker.poi_x == x:
            cont+=1
    print("Result: {} / {}".format(len(pos) - cont, len(pos)))


# Loads NEAT configuration files
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config_sender')
config_send = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    config_path)

config_path = os.path.join(local_dir, 'config_receiver')
config_rec = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    config_path)

population_send = neat.Population(config_send)
population_rec = neat.Population(config_rec)

# Adds statistics report
stats_send = neat.StatisticsReporter()
stats_rec = neat.StatisticsReporter()
population_send.add_reporter(stats_send)
population_send.add_reporter(neat.StdOutReporter(True))
population_send.add_reporter(BestGenomeReporter(name))
population_rec.add_reporter(stats_rec)
population_rec.add_reporter(neat.StdOutReporter(True))
population_rec.add_reporter(BestGenomeReporter(name))

# Parallel execution over all the available processors
#pe = neat.ThreadedEvaluator(multiprocessing.cpu_count(), eval_genome)

# Runs the NEAT algorithm and returns the best network
#winner = population.run(pe.evaluate)

"""
winner_send, winner_rec = run(population_send, population_rec)

pickle_out = open("winner_send.pickle","wb")
pickle.dump(winner_send, pickle_out)
pickle_out.close()
pickle_out = open("winner_rec.pickle","wb")
pickle.dump(winner_rec, pickle_out)
pickle_out.close()
"""




send = pickle.load(open("winner_send.pickle","rb"))
rec = pickle.load(open("winner_rec.pickle","rb"))
viz(send, rec, config_send, config_rec)
