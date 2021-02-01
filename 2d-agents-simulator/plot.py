import graphviz
import matplotlib.pyplot as plt
import numpy as np

NUMBER_OF_GENERATIONS = 30

f = open("exp3.mean", 'r')
stats = f.readlines()
f.close()

gen = range(NUMBER_OF_GENERATIONS)
gen_stats = []
for i in gen:
    gen_stats.append([])

for run in stats:
    run = list(map(float, run.split()))
    for i in range(len(run)):
        gen_stats[i].append(run[i])

mean_gen = []
stddev_gen = []
for i in range(len(gen_stats)):
    mean_gen.append(np.mean(gen_stats[i]))
    stddev_gen.append(np.std(gen_stats[i]))

plt.errorbar(gen, mean_gen, yerr=stddev_gen, label="population's avg")

f = open("exp3.best", 'r')
stats = f.readlines()
f.close()

gen = range(NUMBER_OF_GENERATIONS)
gen_stats = []
for i in gen:
    gen_stats.append([])

for run in stats:
    run = list(map(float, run.split()))
    for i in range(len(run)):
        gen_stats[i].append(run[i])

mean_best_gen = []
stddev_best_gen = []
for i in range(len(gen_stats)):
    mean_best_gen.append(np.mean(gen_stats[i]))
    stddev_best_gen.append(np.std(gen_stats[i]))

plt.errorbar(gen, mean_best_gen, yerr=stddev_best_gen, label="best genome")

plt.title("Navigation task (directions only)")
plt.xlabel("Generations")
plt.xticks(range(NUMBER_OF_GENERATIONS))
plt.ylabel("Population's avg & best genome fitness")
plt.grid()
plt.legend(loc="best")
plt.show()
plt.close()

