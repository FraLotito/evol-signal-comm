import graphviz
import matplotlib.pyplot as plt
import numpy as np

f = open("exp3.mean", 'r')
stats = f.readlines()
f.close()
tmp = []
for i in stats:
    tmp.append(len(i.split()))

x = []
y = []
for i in range(50):
    x.append(i)
    y.append(tmp.count(i))

plt.bar(x, height=y)
plt.yticks(range(1,5))
plt.xticks(range(0, 50, 5))
plt.show()