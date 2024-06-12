# # 4. Visualise Results


# Python imports
import numpy as np

from skimage.transform import rotate

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib as mpl

from tqdm import tqdm
from scipy.optimize import curve_fit

from abel.basex import basex_transform

import tkinter as tk
from tkinter import ttk

# Hyper parameters

# pixels per cm
pxpcm = 367

# Time step between images (min)
dt = 1.

# Molar extinction coefficient (cm^-1 L/mol)
epsilon = 2900.

# Methylene blue molar mass (g mol^-1)
M = 319.85


###########################
# Import previous data
previous_data = np.loadtxt('mass.txt')
time = previous_data[:,0]
mass = previous_data[:,1]
rate = previous_data[:,2]

# Import reconstructed data
reconstructed_data = np.loadtxt('mass_Abel.txt')
mass_Abel = reconstructed_data[:,1]
rate_Abel = reconstructed_data[:,2]

#plot mass vs time
plt.rcParams["figure.figsize"] = (10,8)
plt.plot(time, mass, label='From absorbance, ignoring probe shadow')
plt.plot(time, mass_Abel, label='From concentration field and forward Abel transform')
plt.ylabel('Total mass delivered (µg)')
plt.xlabel('Time (min)')
#plt.ylim(0,22)
plt.legend()
plt.show()

#import concentration images
c = np.load('c_all.npy')

# In[15]:

#### Effect of probe shadow

# floor = np.sum(np.isnan(mask))/(height*width)
# rel_error = 1 - mass/mass_Abel
# plt.plot(time, rel_error, label='Relative error')
# plt.plot([time[0], time[-1]], [floor, floor], label='Shadow')
# plt.show()

# In[16]:
# ### Delivery rates

rate_Abel = np.empty_like(mass_Abel)
rate_Abel[0] = 0
rate_Abel[1:] = np.diff(mass_Abel)/dt

plt.plot(time, rate, label='From absorbance')
plt.plot(time, rate_Abel, label='From concentration and forward Abel transform')
plt.ylabel('Delivery rate (µg/min)')
plt.xlabel('Time (min)')
plt.legend()
plt.grid()
plt.show()


# In[18]:
# ## Interactive Concentration map survey

# Interactive element
# Function to update the plots based on slider values
def update_plots(*args):
    global slider_values, horizontal, vertical 
    slider_values = slider1.get()
    t = slider_values

    #get image

    ax1.clear()
    im1 = ax1.imshow(c[t], cmap=plt.get_cmap('YlGnBu').copy(), vmin=0, vmax=200)
    im1.cmap.set_over('r')
    ax1.set_title('Original concentration map')
    canvas1.draw()

    ax2.clear()
    im2 = ax2.imshow(c[t], cmap=cmap2, norm=norm2)
    im2.cmap.set_over('r')
    ax2.set_title('Discrete concentration map')
    canvas2.draw()
    
    # Update slider value labels
    label1.config(text=f'Slider 1: {slider1.get()}')

# Initialize the main window
root = tk.Tk()
root.title("Concentration Map Survey")

# Create sliders and labels
frame = ttk.Frame(root)
frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

slider1 = tk.Scale(frame, from_=0, to=len(c)-1, orient=tk.VERTICAL, resolution=1,command=update_plots)
slider1.pack()
label1 = tk.Label(frame, text=f'Slider 1: {slider1.get()}')
label1.pack()

# Create the plot area
plot_frame = ttk.Frame(root)
plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

fig = Figure(figsize=(8, 6))

# Plot 1
ax1 = fig.add_subplot(121)
canvas1 = FigureCanvasTkAgg(fig, master=plot_frame)
canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Plot 2
ax2 = fig.add_subplot(122)
canvas2 = FigureCanvasTkAgg(fig, master=plot_frame)
canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
# Build a discrete colorbar
cmap2 = mpl.colors.ListedColormap(["#7fcdbb", "#1d91c0", "#0c2c84"])
cmap2.set_under('#eeeeee')
norm2 = mpl.colors.BoundaryNorm([2, 20, 100, 200], cmap2.N) 



# Initialize the plots with default valuesax1.clear()
t=0
im1 = ax1.imshow(c[t], cmap=plt.get_cmap('YlGnBu').copy(), vmin=0, vmax=200)
im1.cmap.set_over('r')
fig.colorbar(im1, ax=ax1)
ax1.set_title('Original concentration map')
canvas1.draw()

ax2.clear()
im2 = ax2.imshow(c[t], cmap=cmap2, norm=norm2)
im2.cmap.set_over('r')
fig.colorbar(im2, ax=ax2)
ax2.set_title('Discrete concentration map')
canvas2.draw()

update_plots()

# Start the Tkinter main loop
root.mainloop()

########end of GUI pop-up###########
# In[19]:


# Save concentration slice for later use
#np.save('c_20min', c[20])
#np.save('c_60min', c[60])
#np.save('c_120min', c[120])
#np.save('c_5ug', c[42])

