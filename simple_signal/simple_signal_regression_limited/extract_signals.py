import pickle
import os

import neat
import matplotlib.pyplot as plt

MAP_DIM = 5

def plot(send_genome, send_config):
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
    
    for i in signs:
        print(i)

    plt.ylim([0,1])
    for i, s in enumerate(signs):
        plt.plot(s, label=str(i))
    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.legend()
    plt.show()

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config_sender')
config_send = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    config_path)

send = pickle.load(open("winner_send.pickle","rb"))

plot(send, config_send)