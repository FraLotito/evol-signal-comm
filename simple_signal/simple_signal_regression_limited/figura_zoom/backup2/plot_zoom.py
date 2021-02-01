import graphviz
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

NUMBER_OF_GENERATIONS = 26
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams["legend.loc"] = 'upper left'
matplotlib.rcParams.update({'font.size': 15})

f = open("sender.mean", 'r')
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

f = open("sender.best", 'r')
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


#plt.errorbar(gen, mean_best_gen, yerr=stddev_best_gen, label="best genome", fmt='.', capsize=2, barsabove=True, errorevery=10)
#plt.errorbar(gen, mean_gen, yerr=stddev_gen, label="snd population's avg", fmt='.', capsize=2, barsabove=True, errorevery=10)
plt.plot(gen, mean_best_gen, label="Best pair")
plt.plot(gen, mean_gen, label="Sender pop. avg")


f = open("receiver.mean", 'r')
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

f = open("receiver.best", 'r')
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


#plt.errorbar(gen, mean_best_gen, yerr=stddev_best_gen, label="best genome", fmt='.', capsize=2, barsabove=True, errorevery=2)
#plt.errorbar(gen, mean_gen, yerr=stddev_gen, label="rcv population's avg", fmt='.', capsize=1, barsabove=True, errorevery=10)

plt.plot(gen, mean_gen, label="Receiver pop. avg")


plt.xlabel("Generations")
plt.xticks(range(NUMBER_OF_GENERATIONS))
plt.ylabel("Fitness")
locs, labs = plt.xticks()
plt.xticks(locs[::2]) 
#plt.grid()
plt.legend(loc="upper left")
plt.savefig('fitness_trend.pdf')
plt.show()

