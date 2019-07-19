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

PD = Simulation_PD(40, 200, bernoulli_arms, discounted_thompson, 50)
locs,loc_c,loc_d = PD.mult_generation_anim_simple()


# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(-0.5, 1.5), ylim=(-0.5, 1.5))
particles_d, = ax.plot([], [], 'bo', ms=4)
particles_c, = ax.plot([], [], 'go', ms=4)
#parents, = ax.plot([], [], 'ro', ms=4)

# initialization function: plot the background of each frame
def init():
    particles_c.set_data([], [])
    particles_d.set_data([], [])
    #parents.set_data([], [])
    return particles_c,particles_d

# animation function.  This is called sequentially
def animate(i):
    global locs,loc_c,loc_d
    x_c = [z[0] for z in loc_c[i]]
    y_c = [z[1] for z in loc_c[i]]
    particles_c.set_data(x_c,y_c)
    x_d = [z[0] for z in loc_d[i]]
    y_d = [z[1] for z in loc_d[i]]
    particles_d.set_data(x_d,y_d)
    return particles_c,particles_d

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,frames=50, interval=333, blit=True)


anim.save('basic_animation_no_TFT_5_CD.mp4', fps=3, extra_args=['-vcodec', 'libx264'])


plt.show()
