import graphviz
import matplotlib.pyplot as plt
import numpy as np

NUMBER_OF_GENERATIONS = 50
MAX_LEN = 0

f = open("sender.mean", 'r')
lines = f.readlines()
f.close()

sender_mean = []
for line in lines:
    run = list(map(float, line.split()))
    sender_mean.append(run)
    MAX_LEN = max(len(run), MAX_LEN)

print(MAX_LEN)

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

NUMBER_OF_GENERATIONS = MAX_LEN

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

plt.plot(gen, mean_best_gen, color='green', label='best pair avg')
plt.fill_between(gen, [a_i - b_i for a_i, b_i in zip(mean_best_gen, stddev_best_gen)], [a_i + b_i for a_i, b_i in zip(mean_best_gen, stddev_best_gen)], alpha=0.5, color='green')

plt.plot(gen, mean_gen, color='red', label='senders population avg')
plt.fill_between(gen, [a_i - b_i for a_i, b_i in zip(mean_gen, stddev_gen)], [a_i + b_i for a_i, b_i in zip(mean_gen, stddev_gen)], alpha=0.5, color='red')

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

plt.plot(gen, mean_gen, color='blue', label='receivers population avg')
plt.fill_between(gen, [a_i - b_i for a_i, b_i in zip(mean_gen, stddev_gen)], [a_i + b_i for a_i, b_i in zip(mean_gen, stddev_gen)], alpha=0.5, color='blue')


plt.xlabel("Generations")
plt.xticks(range(NUMBER_OF_GENERATIONS))
plt.ylabel("Populations' avg & best pair fitness")
locs, labs = plt.xticks()
plt.xticks(locs[::2000]) 
plt.legend(loc=1)
plt.savefig('fitness_trend.pdf')
plt.show()

