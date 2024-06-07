#!/usr/bin/env python
# coding: utf-8

# # 3. Reconstruct using inverse Abel transform
# 
# In this notebook, we will use absorbance and inverse Abel transform to compute the concentration field assuming axial symmetry.
# 
# First, we should have run the following notebooks:
# - `1 Preprocessing` to extract and denoise absorbance data.
# - `2 Integrate` to measure the mass (minus probe shadow).
# 
# First we import Python modules and define some paraemters.

# In[1]:


# Python imports
import numpy as np

from skimage.transform import rotate

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.optimize import curve_fit

from abel.basex import basex_transform

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

plt.rcParams["figure.figsize"] = (10,8)
plt.plot(time, mass, label='From absorbance, ignoring probe shadow')
plt.plot(time, mass_Abel, label='From concentration field and forward Abel transform')
plt.ylabel('Total mass delivered (µg)')
plt.xlabel('Time (min)')
#plt.ylim(0,22)
plt.legend()
plt.show()

#### Effect of probe shadow

# In[15]:


# floor = np.sum(np.isnan(mask))/(height*width)
# rel_error = 1 - mass/mass_Abel
# plt.plot(time, rel_error, label='Relative error')
# plt.plot([time[0], time[-1]], [floor, floor], label='Shadow')
# plt.show()


# ### Delivery rates

# In[16]:


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


# ## Concentration map survey

# In[18]:


t = 60

plt.rcParams["figure.figsize"] = (15,15)
figure, ax = plt.subplots(ncols=2)

im = ax[0].imshow(c[t], vmin=0, vmax=200)
im.cmap.set_over('r')
figure.colorbar(im, ax=ax[0])
con = ax[1].contourf(c[t], [10, 20, 100, 200], origin='image')
figure.colorbar(con, ax=ax[1])
plt.show()


# In[19]:


# Save concentration slice for later use
#np.save('c_20min', c[20])
#np.save('c_60min', c[60])
#np.save('c_120min', c[120])
#np.save('c_5ug', c[42])


# In[ ]:





# In[ ]:




