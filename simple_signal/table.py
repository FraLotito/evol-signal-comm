N = 20

import numpy as np
a = []

for i in range(N):
    a.append(int(input()))

print("mean: {}".format(np.mean(a)))
print("std: {}".format(np.std(a)))