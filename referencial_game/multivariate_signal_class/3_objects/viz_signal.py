import pickle
import os

import neat
import matplotlib.pyplot as plt

NUM_OGGETTI = 3

oggetti = [[0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 0], [0, 0, 0], [1, 1, 1], [0, 0, 1], [1, 0, 0]]
oggetti = oggetti[:NUM_OGGETTI]

def plot(send_genome, send_config):
    net_send = neat.ctrnn.CTRNN.create(send_genome, send_config, 1)

    signs = []

    for y in oggetti:
        dt = 1
        t = 0
        T_MAX = 10
        #norm_y = y / MAP_DIM

        net_send.reset()
        sign = []
        while t < T_MAX:
            t += dt
            out1 = net_send.advance(y, dt, dt)
            sign.extend(out1)

        signs.append(sign)
    
    #for i in signs:
    #    print(i)

    for i, s in enumerate(signs):
        plt.plot(s, label=str(i))
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.legend()
    plt.show()
    

def write_signal(send_genome, send_config, gen):
    net_send = neat.ctrnn.CTRNN.create(send_genome, send_config, 1)

    signs = []

    for y in range(MAP_DIM):
        dt = 1
        t = 0
        T_MAX = 10
        norm_y = y / MAP_DIM

        net_send.reset()
        sign = []
        while t < T_MAX:
            t += dt
            out1 = net_send.advance([norm_y], dt, dt)
            sign.extend(out1)

        signs.append(sign)
    fopen = open("{}_signal".format(gen), 'a+')
    for i in signs:
        fopen.write(" ".join(list(map(str, i))))
        fopen.write('\n')


local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config_sender')
config_send = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    config_path)

def extract_gen(f):
    insert = False
    res = ""
    for i in f:
        if i == '.':
            insert = False
        if insert:
            res += i
        if i == '_':
            insert = True
    return res

def extract_id(f):
    insert = False
    res = ""
    for i in f:
        if insert:
            res += i
        if i == '_':
            insert = True
    return res

for f in os.listdir():
    if 'exp' in f:
        name = "winner_send_" + extract_id(f) + ".pickle"
        send = pickle.load(open(f + '//' + name,"rb"))
        #write_signal(send, config_send, extract_gen(f))
        plot(send, config_send)
    