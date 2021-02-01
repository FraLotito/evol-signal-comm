import graphviz
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams["legend.loc"] = 'upper left'
matplotlib.rcParams.update({'font.size': 22})
plt.rcParams["figure.figsize"] = (6.4, 4.8)

NUMBER_OF_GENERATIONS = 50
MAX_LEN = 0
NTH = 50

f = open("sender.mean", 'r')
lines = f.readlines()
f.close()

sender_mean = []
for line in lines:
    run = list(map(float, line.split()))
    sender_mean.append(run)
    MAX_LEN = max(len(run), MAX_LEN)

f = open("receiver.mean", 'r')
lines = f.readlines()
f.close()

receiver_mean = []
for line in lines:
    run = list(map(float, line.split()))
    receiver_mean.append(run)

f = open("receiver.best", 'r')
lines = f.readlines()
f.close()

receiver_best = []
for line in lines:
    run = list(map(float, line.split()))
    receiver_best.append(run)

NUMBER_OF_GENERATIONS = MAX_LEN + NTH

for i in sender_mean:
    while len(i) < NUMBER_OF_GENERATIONS:
        i.append(i[-1])

for i in receiver_mean:
    while len(i) < NUMBER_OF_GENERATIONS:
        i.append(i[-1])

for i in receiver_best:
    while len(i) < NUMBER_OF_GENERATIONS:
        i.append(i[-1])


gen = range(NUMBER_OF_GENERATIONS)
gen_stats = []
for i in gen:
    gen_stats.append([])

for run in sender_mean:
    for i in range(len(run)):
        gen_stats[i].append(run[i])

mean_gen = []
stddev_gen = []
for i in range(len(gen_stats)):
    mean_gen.append(np.mean(gen_stats[i]))
    stddev_gen.append(np.std(gen_stats[i]))

gen = range(NUMBER_OF_GENERATIONS)
gen_stats = []
for i in gen:
    gen_stats.append([])

for run in receiver_best:
    for i in range(len(run)):
        gen_stats[i].append(run[i])

mean_best_gen = []
stddev_best_gen = []
for i in range(len(gen_stats)):
    mean_best_gen.append(np.mean(gen_stats[i]))
    stddev_best_gen.append(np.std(gen_stats[i]))

plt.plot(gen[::NTH], mean_best_gen[::NTH], color='green', label='best pair avg')
plt.fill_between(gen[::NTH], [a_i - b_i for a_i, b_i in zip(mean_best_gen[::NTH], stddev_best_gen[::NTH])], [a_i + b_i for a_i, b_i in zip(mean_best_gen[::NTH], stddev_best_gen[::NTH])], alpha=0.25, color='green')

plt.plot(gen[::NTH], mean_gen[::NTH], color='red', label='senders population avg')
plt.fill_between(gen[::NTH], [a_i - b_i for a_i, b_i in zip(mean_gen[::NTH], stddev_gen[::NTH])], [a_i + b_i for a_i, b_i in zip(mean_gen[::NTH], stddev_gen[::NTH])], alpha=0.25, color='red')

gen = range(NUMBER_OF_GENERATIONS)
gen_stats = []
for i in gen:
    gen_stats.append([])

for run in receiver_mean:
    for i in range(len(run)):
        gen_stats[i].append(run[i])

mean_gen = []
stddev_gen = []
for i in range(len(gen_stats)):
    mean_gen.append(np.mean(gen_stats[i]))
    stddev_gen.append(np.std(gen_stats[i]))

plt.plot(gen[::NTH], mean_gen[::NTH], color='blue', label='receivers population avg')
plt.fill_between(gen[::NTH], [a_i - b_i for a_i, b_i in zip(mean_gen[::NTH], stddev_gen[::NTH])], [a_i + b_i for a_i, b_i in zip(mean_gen[::NTH], stddev_gen[::NTH])], alpha=0.25, color='blue')

print(NUMBER_OF_GENERATIONS)

plt.xlabel("Generations")
plt.xticks(range(NUMBER_OF_GENERATIONS))
plt.ylabel("Fitness")
locs, labs = plt.xticks()
plt.xticks(locs[::2500]) 
#plt.legend(loc=0)
plt.savefig('fitness_trend_classification_unlimited.pdf', bbox_inches = 'tight')
plt.show()

