import graphviz
import matplotlib.pyplot as plt
import numpy as np

NUMBER_OF_GENERATIONS = 200

f = open("sender.mean", 'r')
stats = f.readlines()
f.close()

gen = range(NUMBER_OF_GENERATIONS)
gen_stats = []
for i in gen:
    gen_stats.append([])

for run in stats:
    run = list(map(float, run.split()))
    run = run[:NUMBER_OF_GENERATIONS]
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
    run = run[:NUMBER_OF_GENERATIONS]

    for i in range(len(run)):
        gen_stats[i].append(run[i])

mean_best_gen = []
stddev_best_gen = []
for i in range(len(gen_stats)):
    mean_best_gen.append(np.mean(gen_stats[i]))
    stddev_best_gen.append(np.std(gen_stats[i]))

plt.plot(gen, mean_best_gen, color='green', label='best pair avg')
plt.fill_between(gen, [a_i - b_i for a_i, b_i in zip(mean_best_gen, stddev_best_gen)], [a_i + b_i for a_i, b_i in zip(mean_best_gen, stddev_best_gen)], alpha=0.35, color='green')

plt.plot(gen, mean_gen, color='red', label='senders population avg')
plt.fill_between(gen, [a_i - b_i for a_i, b_i in zip(mean_gen, stddev_gen)], [a_i + b_i for a_i, b_i in zip(mean_gen, stddev_gen)], alpha=0.35, color='red')

#plt.errorbar(gen, mean_best_gen, yerr=stddev_best_gen, label="best genome", fmt='.', capsize=2, barsabove=True, errorevery=2)
#plt.errorbar(gen, mean_gen, yerr=stddev_gen, label="snd population's avg", fmt='.', capsize=2, barsabove=True, errorevery=2)

f = open("receiver.mean", 'r')
stats = f.readlines()
f.close()

gen = range(NUMBER_OF_GENERATIONS)
gen_stats = []
for i in gen:
    gen_stats.append([])

for run in stats:
    run = list(map(float, run.split()))
    run = run[:NUMBER_OF_GENERATIONS]

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
    run = run[:NUMBER_OF_GENERATIONS]

    for i in range(len(run)):
        gen_stats[i].append(run[i])

mean_best_gen = []
stddev_best_gen = []
for i in range(len(gen_stats)):
    mean_best_gen.append(np.mean(gen_stats[i]))
    stddev_best_gen.append(np.std(gen_stats[i]))

plt.plot(gen, mean_gen, color='blue', label='receivers population avg')
plt.fill_between(gen, [a_i - b_i for a_i, b_i in zip(mean_gen, stddev_gen)], [a_i + b_i for a_i, b_i in zip(mean_gen, stddev_gen)], alpha=0.35, color='blue')
#plt.errorbar(gen, mean_gen, yerr=stddev_gen, label="rcv population's avg", fmt='.', capsize=2, barsabove=True, errorevery=2)


plt.xlabel("Generations")
plt.xticks(range(NUMBER_OF_GENERATIONS + 1))
plt.ylabel("Populations' avg & best pair fitness")
locs, labs = plt.xticks()
plt.xticks(locs[::25]) 
plt.legend(loc="best")
plt.savefig('fitness_data_reg_limit_5s.pdf')
plt.show()

