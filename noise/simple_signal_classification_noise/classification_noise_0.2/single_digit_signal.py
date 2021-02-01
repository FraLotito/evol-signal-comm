import numpy as np
import neat
import os
import multiprocessing
import random
import pickle
from neat.reporting import BaseReporter
from neat.math_util import softmax as softmax1


name = str(random.randint(0, 9999999))
dir_name = "exp_{}".format(name)
os.mkdir(dir_name)
SUP_LIM = 5
MAP_DIM = 5
SENDER_REPORT = None
LOGGING = True
SIGMA = 0.2
MEAN_ERROR = 0

def manhattan_distance(a,b):
    return abs(b[0]-a[0]) + abs(b[1]-a[1])

def cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector. 
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    X = np.array([X])
    m = y.shape[0]
    p = softmax(X)
    # We use multidimensional array indexing to extract 
    # softmax probability of the correct label for each sample.
    # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
    log_likelihood = -np.log(p[range(m),y])
    loss = np.sum(log_likelihood) / m
    return loss


class BestGenomeReporter(BaseReporter):
    """Uses `print` to output information about the run; an example reporter class."""
    def __init__(self, name):
        #self.show_species_detail = show_species_detail
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0
        self.store_mean_sender = []
        self.store_best_sender = []
        self.store_mean_receiver = []
        self.store_best_receiver = []
        self.name = name

    def post_evaluate(self, config, population, species, best_genome):
        global SENDER_REPORT
        fitnesses = [c.fitness for c in population.values()]
        fit_mean = np.mean(fitnesses)
        if SENDER_REPORT:
            self.store_mean_sender.append(fit_mean)
            self.store_best_sender.append(best_genome.fitness)
        else:
            self.store_mean_receiver.append(fit_mean)
            self.store_best_receiver.append(best_genome.fitness)

    def found_solution(self, config, generation, best):
        global SENDER_REPORT
        
        if SENDER_REPORT and LOGGING:
            f = open('sender.mean', 'a+')
            f.write(' '.join(list(map(str, self.store_mean_sender))) + '\n')
            f.close()
            f = open('sender.best', 'a+')
            f.write(' '.join(list(map(str, self.store_best_sender))) + '\n')
            f.close()
        else:
            f = open('receiver.mean', 'a+')
            f.write(' '.join(list(map(str, self.store_mean_receiver))) + '\n')
            f.close()
            f = open('receiver.best', 'a+')
            f.write(' '.join(list(map(str, self.store_best_receiver))) + '\n')
            f.close()

            #f = open('complexity_MAPDIM_{}'.format(MAP_DIM), 'a+')
            #f.write(str(generation) + '\n')
            #f.close()

        

class NeuralAttacker():
    def __init__(self, genome, config):
        self.net_rec = neat.ctrnn.CTRNN.create(genome, config, 1)
        self.action = []

    def listen(self, inputs, dt):
        self.out = self.net_rec.advance(inputs, dt, dt)
    
    def learn(self):
        self.softmax_result = softmax1(self.out)
        self.poi_y = np.argmax(((self.softmax_result / np.max(self.softmax_result)) == 1).astype(int))


class CompleteExtinctionException(Exception):
    pass

results = {}

sending = [i for i in range(SUP_LIM)]
random.shuffle(sending)
sending = sending[:MAP_DIM]

def eval_genomes(genomes, send_config, rec_config):
    results.clear()

    for genome in genomes:
        id_send = genome[0][0]
        id_rcv = genome[1][0]
        send = genome[0][1]
        rec = genome[1][1]
        net_send = neat.ctrnn.CTRNN.create(send, send_config, 1)
        receiver = NeuralAttacker(rec, rec_config)

        fitnesses = []

        for Y in range(MAP_DIM * 3):
            y = Y % MAP_DIM
            dt = 1
            t = 0
            T_MAX = 10
            receiver.net_rec.reset()
            net_send.reset()

            norm_y = y / MAP_DIM

            while t < T_MAX:
                t += dt
                out1 = net_send.advance([norm_y], dt, dt)
                out1[0] += random.gauss(MEAN_ERROR, SIGMA)
                receiver.listen(out1, dt)
            
            receiver.learn()  

            if y != receiver.poi_y:
                fitnesses.append(-1)

        results[(id_send, id_rcv)] = sum(fitnesses)

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

        #print(results)

        best_v = None
        best_k = None

        #rcv_fitness = [None]*(n2 + 1)
        #send_fitness = [None]*(n1 + 1)

        rcv_fitness = {}
        send_fitness = {}

        for key in results:
            if best_v is None:
                best_v = results[key]
                best_k = key
            else:
                if results[key] > best_v:
                    best_v = results[key]
                    best_k = key

            if key[1] not in rcv_fitness:
                rcv_fitness[key[1]] = results[key]
            else:
                rcv_fitness[key[1]] = max(rcv_fitness[key[1]], results[key])

            if key[0] not in send_fitness:
                send_fitness[key[0]] = results[key]
            else:
                send_fitness[key[0]] = max(send_fitness[key[0]], results[key])

        best_send_id = best_k[0]
        best_rcv_id = best_k[1]
        #print("BEST VALUE: {}, id: {}".format(best_v, best_k))
        

        # Gather and report statistics.
        best_send = None
        best_rec = None
        for g in population_send.population.values():
            g.fitness = send_fitness[g.key]
            #raise RuntimeError("Fitness not assigned to genome {}".format(g.key))
            if g.key == best_send_id:
                best_send = g
        #print(_id)
        for g in population_rec.population.values():
            g.fitness = rcv_fitness[g.key]
            if g.key == best_rcv_id:
                best_rec = g
                #_id2 = g.key
        #print(_id2)
        #exit()
        global SENDER_REPORT
        SENDER_REPORT = True
        population_send.reporters.post_evaluate(population_send.config, population_send.population, population_send.species, best_send)
        SENDER_REPORT = False
        population_rec.reporters.post_evaluate(population_rec.config, population_rec.population, population_rec.species, best_rec)


        # Track the best genome ever seen.
        #if population_send.best_genome is None or best_send.fitness > population_send.best_genome.fitness:
        population_send.best_genome = best_send
        population_rec.best_genome = best_rec
        #print("BEST POPULATION: {}, {}, id: {}".format(best_send.fitness, best_rec.fitness, (best_send.key, best_rec.key)))
        
        #if population_rec.best_genome is None or best_rec.fitness > population_rec.best_genome.fitness:
            #population_rec.best_genome = best_rec

        # this is true for both since the fitness is shared
        if not population_send.config.no_fitness_termination:
            # End if the fitness threshold is reached.
            fv = population_send.fitness_criterion(g.fitness for g in population_send.population.values())
            if fv >= population_send.config.fitness_threshold:
                SENDER_REPORT = True
                population_send.reporters.found_solution(population_send.config, population_send.generation, best_send)
                SENDER_REPORT = False
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
        SENDER_REPORT = True
        population_send.reporters.found_solution(population_send.config, population_send.generation, population_send.best_genome)
        SENDER_REPORT = False
        population_rec.reporters.found_solution(population_rec.config, population_rec.generation, population_rec.best_genome)

    return population_send.best_genome, population_rec.best_genome

def viz(send_genome, rec_genome, send_config, rec_config):
    net_send = neat.ctrnn.CTRNN.create(send_genome, send_config, 1)
    receiver = NeuralAttacker(rec_genome, rec_config)

    cont = 0
    for Y in range(100):
        #y, x = random.randint(0, MAP_DIM + 1), random.randint(0, MAP_DIM + 1)
        #print("TRUE: {}".format(y))
        y = Y % MAP_DIM
        dt = 1
        t = 0
        T_MAX = 10
        receiver.net_rec.reset()
        net_send.reset()
        sign = []
        norm_y = y / MAP_DIM
        while t < T_MAX:
            t += dt
            out1 = net_send.advance([norm_y], dt, dt)
            out1[0] += random.gauss(MEAN_ERROR, SIGMA)
            sign.append(out1)
            receiver.listen(out1, dt)
        receiver.learn()  
        #print(sign)
        #print("PRED: {}".format(receiver.poi_y))
        if receiver.poi_y == y: #and attacker.poi_x == x:
            cont+=1
    
    fopen = open("test_noise_{}".format(SIGMA), 'a+')
    fopen.write(str(cont))
    fopen.write('\n')
    fopen.close()

    #print("Result: {} / {}".format(cont, SUP_LIM))

def plot(send_genome, rec_genome, send_config, rec_config):
    import random
    name = random.randint(0, 2000)
    fopen = open("{}_data".format(name), 'w')

    net_send = neat.ctrnn.CTRNN.create(send_genome, send_config, 1)
    receiver = NeuralAttacker(rec_genome, rec_config)
    signs = []
    cont = 0
    for y in range(MAP_DIM):
        #y, x = random.randint(0, MAP_DIM + 1), random.randint(0, MAP_DIM + 1)
        print("TRUE: {}".format(y))
        dt = 1
        t = 0
        T_MAX = 10
        receiver.net_rec.reset()
        net_send.reset()
        sign = []
        norm_y = y / MAP_DIM
        while t < T_MAX:
            t += dt
            out1 = net_send.advance([norm_y], dt, dt)
            sign.append(out1)
            fopen.write(str(out1[0]) + ' ')
            receiver.listen(out1, dt)
        receiver.learn()  
        print(sign)
        signs.append(sign)
        fopen.write('\n')
        print("PRED: {}".format(receiver.poi_y))
        if receiver.poi_y == y: #and attacker.poi_x == x:
            cont+=1

    print("Result: {} / {}".format(cont, MAP_DIM))
    fopen.close()

    import matplotlib.pyplot as plt
    
    for i, s in enumerate(signs):
        plt.plot(s, label=str(i))
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.legend()
    plt.savefig('{}_plot.pdf'.format(name))


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
population_rec.add_reporter(BestGenomeReporter(name))
population_send.add_reporter(BestGenomeReporter(name))

stats_send = neat.StatisticsReporter()
stats_rec = neat.StatisticsReporter()
population_send.add_reporter(stats_send)
population_send.add_reporter(neat.StdOutReporter(True))
population_rec.add_reporter(stats_rec)
population_rec.add_reporter(neat.StdOutReporter(True))


winner_send, winner_rec = run(population_send, population_rec)

pickle_out = open("{}//winner_send_{}.pickle".format(dir_name, name),"wb")
pickle.dump(winner_send, pickle_out)
pickle_out.close()
pickle_out = open("{}//winner_rec_{}.pickle".format(dir_name, name),"wb")
pickle.dump(winner_rec, pickle_out)
pickle_out.close()

pickle_out = open("{}//stats_send_{}.pickle".format(dir_name, name),"wb")
pickle.dump(stats_send, pickle_out)
pickle_out.close()

pickle_out = open("{}//stats_rec_{}.pickle".format(dir_name, name),"wb")
pickle.dump(stats_rec, pickle_out)
pickle_out.close()


#send = pickle.load(open("winner_send.pickle","rb"))
#rec = pickle.load(open("winner_rec.pickle","rb"))

#plot(send, rec, config_send, config_rec)
viz(winner_send, winner_rec, config_send, config_rec)
