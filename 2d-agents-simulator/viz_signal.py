import matplotlib.pyplot as plt
import matplotlib.animation as animation
from celluloid import Camera
from matplotlib.ticker import FormatStrFormatter

N_samples = 4
fig, a =  plt.subplots(N_samples)
camera = Camera(fig)

fopen = open("signals", 'r')
lines = fopen.readlines()
fopen.close()

generation = []

gen_cont = 0


for line in lines:
    if len(generation) == N_samples:
        generation = []
    
    line = list(map(float, line.split()))
    generation.append(line)

    if len(generation) == N_samples:
        if gen_cont % 5 == 0:
            #for i in range(N_samples):
                #a[i].clear()
            for i in range(len(generation)):
                data = generation[i]
                a[i].plot(data, c='b')
                #a[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            fig.canvas.draw()
            camera.snap()
        gen_cont += 1
        print('*'*30)
        print(generation)

for i in range(len(generation)):
    data = generation[i]
    a[i].plot(data, c='b')
    #a[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
fig.canvas.draw()
camera.snap()
        
print(len(lines))

animation = camera.animate() 
animation.save('subplots.mp4')
#anim = animation.FuncAnimation(fig, animate, interval=100)
#plt.show()