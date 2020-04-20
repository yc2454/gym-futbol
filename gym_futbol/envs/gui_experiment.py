import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
import random
import time

fig, ax = plt.subplots()
xdata, ydata = [], []
field, = plt.plot([], [], 'ro')

players = [] # holds players' coordinates

def init():
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    for i in range(10):
        players.append([random.random() * 1000, random.random() * 1000])
    return field,
    
    
# Return x in (x,y)
def x_coor(coors):
    return coors[0]

# Return y in (x,y)
def y_coor(coors):
    return coors[1]

def players_step():
    for player in players:
        player[0] += 100
        player[1] -= 50
        
        if player[0] > 1000:
            player[0] = random.random() * 1000
        if player[1] > 1000:
            player[1] = random.random() * 1000

def animate(i):
    """perform animation step"""
    players_step()
    time.sleep(5)
    xdata = list(map(x_coor, players))
    ydata = list(map(y_coor, players))
#    field.set_data(map(x_coor, players), map(y_coor, players)) # map (higher order function)
    field.set_data(xdata, ydata)
    return field,
        

ani = animation.FuncAnimation(fig, animate, frames=600,
                              interval=10, blit=True, init_func=init)


# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#ani.save('particle_box.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
