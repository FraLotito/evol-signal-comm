import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams["legend.loc"] = 'upper left'
matplotlib.rcParams.update({'font.size': 22})
plt.rcParams["figure.figsize"] = (6.4, 4.8)

fopen = open("complexity_data", 'r')

data = fopen.readlines()

fopen.close()

d = {}
std_dev = {}
cont = 0
for line in data:
    line = line.strip()
    if len(line) == 0:
        continue
    if cont % 11 == 0:
        key = line
        d[key] = []
        std_dev[key] = []
    else:
        line = line.split()
        line = list(map(int, line))
        #print(line)
        d[key].append(sum(line) / len(line))
        std_dev[key].append(np.std(line))
    cont += 1

fig, ax = plt.subplots()
plt.subplots_adjust(left=.2)

ax.set(xlabel='Vocabulary size', ylabel='Generations to converge')

#ax.set_ylim([0,10000])
ax.set_xticks(range(10))
ax.set_xticklabels(range(1, 11))
#ax.set_yticks(np.arange(0, 10000+1, 1000))

UP_LIM_CLASS = 5
UP_UNLIM_CLASS = 7
LINEWIDTH = 1

#print(d['Limited classification'])
#print(d['Unlimited classification'])

d['Limited classification'] = d['Limited classification'][:UP_LIM_CLASS]
d['Unlimited classification'] = d['Unlimited classification'][:UP_UNLIM_CLASS]
std_dev['Limited classification'] = std_dev['Limited classification'][:UP_LIM_CLASS]
std_dev['Unlimited classification'] = std_dev['Unlimited classification'][:UP_UNLIM_CLASS]

#plt.yscale('symlog')

ax.plot(d['Unlimited regression'], label='Unlimited', color='orange', linewidth=LINEWIDTH)
ax.fill_between(range(10), [a_i - b_i for a_i, b_i in zip(d['Unlimited regression'], std_dev['Unlimited regression'])], [a_i + b_i for a_i, b_i in zip(d['Unlimited regression'], std_dev['Unlimited regression'])], alpha=0.25, color='orange')
y = np.array(d['Unlimited regression'])
yerr = np.array(std_dev['Unlimited regression'])
#yerr = 0.434 * yerr
#yerr = np.divide(yerr, y + 1e-10)
#yerr[yerr>=y] = y[yerr>=y]*.999999

#ax.errorbar(range(10), y, yerr=yerr, color='orange', barsabove=True, fmt='.', capthick=2, elinewidth=1)


ax.plot(d['Limited regression'], label='Limited', color='blue', linewidth=LINEWIDTH)
ax.fill_between(range(10), [a_i - b_i for a_i, b_i in zip(d['Limited regression'], std_dev['Limited regression'])], [a_i + b_i for a_i, b_i in zip(d['Limited regression'], std_dev['Limited regression'])], alpha=0.25, color='blue')
y = np.array(d['Limited regression'])
yerr = np.array(std_dev['Limited regression'])
#yerr = 0.434 * yerr
#yerr = np.divide(yerr, y + 1e-10)

#ax.errorbar(range(10), y, yerr=yerr, color='blue', barsabove=True, fmt='.', capthick=2, elinewidth=1)

#yerr = 0.434 * yerr
#yerr = np.divide(yerr, y + 1e-10)
#ax.errorbar(range(UP_LIM_CLASS), y, yerr=yerr, color='green', barsabove=True, fmt='.', capthick=2, elinewidth=1)


plt.legend()
plt.savefig("reg_complexity.pdf", bbox_inches = 'tight')