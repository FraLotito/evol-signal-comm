import graphviz
import matplotlib.pyplot as plt
import numpy as np

NUMBER_OF_GENERATIONS = -1

f = open("exp3.mean", 'r')
stats = f.readlines()
f.close()

for run in stats:
    run = list(map(float, run.split()))
    NUMBER_OF_GENERATIONS = max(NUMBER_OF_GENERATIONS, len(run))

gen = range(NUMBER_OF_GENERATIONS)
gen_stats = []
for i in gen:
    gen_stats.append([])

for run in stats:
    run = list(map(float, run.split()))
    while len(run) < NUMBER_OF_GENERATIONS:
        run.append(run[-1])
    for i in range(len(run)):
        gen_stats[i].append(run[i])

mean_gen = []
stddev_gen = []
for i in range(len(gen_stats)):
    mean_gen.append(np.mean(gen_stats[i]))
    stddev_gen.append(np.std(gen_stats[i]))

f = open("exp3.best", 'r')
stats = f.readlines()
f.close()

gen = range(NUMBER_OF_GENERATIONS)
gen_stats = []
for i in gen:
    gen_stats.append([])

for run in stats:
    run = list(map(float, run.split()))
    while len(run) < NUMBER_OF_GENERATIONS:
        run.append(run[-1])
    #run = run[:NUMBER_OF_GENERATIONS]

    for i in range(len(run)):
        gen_stats[i].append(run[i])

mean_best_gen = []
stddev_best_gen = []
for i in range(len(gen_stats)):
    mean_best_gen.append(np.mean(gen_stats[i]))
    stddev_best_gen.append(np.std(gen_stats[i]))

plt.plot(gen, mean_best_gen, color='green', label='best genome avg fitness')
plt.fill_between(gen, [a_i - b_i for a_i, b_i in zip(mean_best_gen, stddev_best_gen)], [a_i + b_i for a_i, b_i in zip(mean_best_gen, stddev_best_gen)], alpha=0.35, color='green')

plt.plot(gen, mean_gen, color='orange', label='population avg fitness')
plt.fill_between(gen, [a_i - b_i for a_i, b_i in zip(mean_gen, stddev_gen)], [a_i + b_i for a_i, b_i in zip(mean_gen, stddev_gen)], alpha=0.35, color='orange')


plt.xlabel("Generations")
plt.xticks(range(NUMBER_OF_GENERATIONS))
plt.ylabel("Populations' avg & best genome fitness")
locs, labs = plt.xticks()
#plt.xticks(locs[::25]) 
plt.legend(loc="best")
plt.savefig('fitness_data_navigation.pdf')
plt.show()

