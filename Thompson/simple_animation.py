# -*- coding: utf-8 -*-
"""
Animation of the movement behavior over generations
"""




import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


plt.rcParams['animation.ffmpeg_path'] = 'C:\\FFmpeg\\bin\\ffmpeg.exe'
FFwriter = animation.FFMpegWriter()


from evolving_comm_network import *

PD = Simulation_PD(50, 250, bernoulli_arms, discounted_thompson, 100)
locs, locs_parents, mean_comm_coop, len_comm_coop,mean_comm_spatial,len_comm_spatial = PD.mult_generation_anim()


# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
particles, = ax.plot([], [], 'bo', ms=4)
parents, = ax.plot([], [], 'ro', ms=4)

# initialization function: plot the background of each frame
def init():
    particles.set_data([], [])
    #parents.set_data([], [])
    return particles,

# animation function.  This is called sequentially
def animate(i):
    global locs, locs_parents
    x = [z[0] for z in locs[i]]
    y = [z[1] for z in locs[i]]
    particles.set_data(x,y)
    return particles,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=100, interval=333, blit=True)


anim.save('basic_animation.mp4', fps=3, extra_args=['-vcodec', 'libx264'])


plt.show()
