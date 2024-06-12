#!/usr/bin/env python
# coding: utf-8

# # 1. Preprocessing
# 
# In this notebook, we apply some preprocessing steps to our image stack:
# 1. Import and crop.
# 2. Denoise time-wise.
# 3. Calculate absorbance and denoise space-wise.
# 4. Save denoised absorbance as double-precision array stack to a file.
# 
# First we import Python modules and define some paraemters.

# In[1]:


# Python imports
import os
from natsort import natsorted, ns

import numpy as np

from skimage import io
from skimage.util import img_as_float
from skimage.restoration import denoise_wavelet, estimate_sigma

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ipywidgets import interact, interactive, fixed, interact_manual, Layout
import ipywidgets as widgets

from tqdm.notebook import tqdm

import tkinter as tk
from tkinter import ttk

# In[2]:


# folder containing the images. This should be the relative path from the python
# script.
folder = 'images'

# Reference image Python index (for example, if reference image is the first of the stack,
# leave as 0)
ref_index = 0

# Last image to take (None to take the last image)
ref_end = None

# Number of images to skip (use non-zero for testing or to speed-up processing)
skip = 0


# In[4]:


# Get images names. 'natsorted' function will sort the names by 'natural' order,
# for example if there is a suffix with image number or timestamp.
list_files = natsorted(os.listdir(folder))

# Trim list
list_files = list_files[ref_index:ref_end:skip+1]
print(f"Reference image name: {list_files[0]}")
print(f"Number of images in stack: {len(list_files)}")


# ## Select the area of interest
# 
# Use slider handles or select and overwrite values next to each the slider.

# In[5]:


# Cropping
im0 = img_as_float(io.imread(f"{folder}/{list_files[0]}"))
global y_max, x_max
y_max, x_max = im0.shape

# Interactive element
# Function to update the plots based on slider values
def crop_update_plots(*args):
    global slider_values, horizontal, vertical 
    slider_values = [slider1.get(), slider2.get(), slider3.get(), slider4.get()]
    horizontal = [slider_values[0], x_max-slider_values[1]]
    vertical = [slider_values[2], y_max-slider_values[3]]
    #print(horizontal)
    #print(vertical)
    
    #get image
    im = im0[vertical[0]:vertical[1],horizontal[0]:horizontal[1]]
    rect = patches.Rectangle([horizontal[0], vertical[0]], horizontal[1]-horizontal[0], vertical[1]-vertical[0], linewidth=1, edgecolor='r', facecolor='none')
    
    ax1.clear()
    ax1.imshow(im0, cmap='gray')
    ax1.add_patch(rect)
    ax1.set_title('Original image')
    canvas1.draw()

    ax2.clear()
    ax2.imshow(im, cmap='gray')
    ax2.set_title('Cropped image')
    canvas2.draw()
    
    # Update slider value labels
    label1.config(text=f'Slider 1: {slider1.get()}')
    label2.config(text=f'Slider 2: {slider2.get()}')
    label3.config(text=f'Slider 3: {slider3.get()}')
    label4.config(text=f'Slider 4: {slider4.get()}')



# Initialize the main window
root = tk.Tk()
root.title("Crop images to gel of interest only")
root.geometry("600x600")

# Create sliders and labels
frame = ttk.Frame(root)
frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

slider1 = tk.Scale(frame, from_=0, to=x_max/2, orient=tk.HORIZONTAL, resolution=1,command=crop_update_plots)
slider1.pack()
label1 = tk.Label(frame, text=f'Slider 1: {slider1.get()}')
label1.pack()

slider2 = tk.Scale(frame, from_=0, to=x_max/2, orient=tk.HORIZONTAL, resolution=1,command=crop_update_plots)
slider2.pack()
label2 = tk.Label(frame, text=f'Slider 2: {slider2.get()}')
label2.pack()

slider3 = tk.Scale(frame, from_=0, to=y_max/2, orient=tk.HORIZONTAL, resolution=1,command=crop_update_plots)
slider3.pack()
label3 = tk.Label(frame, text=f'Slider 3: {slider3.get()}')
label3.pack()

slider4 = tk.Scale(frame, from_=0, to=y_max/2, orient=tk.HORIZONTAL, resolution=1,command=crop_update_plots)
slider4.pack()
label4 = tk.Label(frame, text=f'Slider 4: {slider4.get()}')
label4.pack()

# Create the plot area
plot_frame = ttk.Frame(root)
plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

fig = Figure(figsize=(8, 6))

# Plot 1
ax1 = fig.add_subplot(211)
canvas1 = FigureCanvasTkAgg(fig, master=plot_frame)
canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Plot 2
ax2 = fig.add_subplot(212)
canvas2 = FigureCanvasTkAgg(fig, master=plot_frame)
canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


# Initialize the plots with default values
crop_update_plots()

# Start the Tkinter main loop
root.mainloop()


print(slider_values)

########end of GUI pop-up###########


# In[6]:



# ## Load images
# 
# Images of the stack will be loaded, cropped and stored in an array of images for further processing.

# In[7]:
print('start loading images')

stack = np.empty((len(list_files), *[vertical[1]-vertical[0],horizontal[1]-horizontal[0]]))

for i, name in tqdm(enumerate(list_files), total=len(list_files)):

    # Load image as matrix of floats, values in [0,1]
    im = img_as_float(io.imread(f"{folder}/{name}"))
    # Crop and store
    stack[i] = im[vertical[0]:vertical[1],horizontal[0]:horizontal[1]]

print('finish loading images')

# ## Time noise reduction using wavelets
# 
# The following noise reduction is based on the fact that for each pixel the intensity should be either constant (outside diffusion area) or a smooth function of time. We apply wavelet denoising with a standard algorithm from the `skimage.restoration` module. Parameters are stored in the following dictionary:

# In[8]:


denoise_kwargs = {'wavelet': 'sym4', 
                  'mode': 'soft',
                  'method': 'VisuShrink'
                  }


# In[10]:


# Overwrite denoise_wavelet to force strong denoising
import pywt

noise_floor = 0.05

def denoise_wavelet(s, sigma=0, **denoise_kwargs):
    coeffs = pywt.wavedec(s, 'sym3', mode='symmetric')
    for i in range(1,len(coeffs)):
        coeffs[i][np.abs(coeffs[i])<noise_floor] = 0
    reconstruct = pywt.waverec(coeffs, 'sym3', mode='symmetric')  
    if len(reconstruct) != len(s):
        return reconstruct[:-1]
    else:
        return reconstruct
    


# First we try the filtering on a few point probes. The standard deviation of the noise is computed using `estimate_sigma`.

# In[11]:

global height, width
# Number of probes in vertical direction
Nh = 1
# Number of probes in horizontal direction
Nw = 6
height, width = stack[0].shape
h_base = np.arange(height//(2*Nh), height-height//(2*Nh)+1, height//(Nh), dtype=int)
w_base = np.arange(width//(2*Nw), width-width//(2*Nw)+1, width//(Nw), dtype=int)

ws, hs = np.meshgrid(w_base, h_base)
ws, hs = ws.flatten(), hs.flatten()

plt.imshow(stack[0])
plt.plot(ws,hs,'r+')
plt.show()

probe_signal = np.empty((Nh*Nw, stack.shape[0]))
sigmas = np.empty(Nh*Nw)
# First pass to estimate the noise standard deviation
for i in range(Nh*Nw):
    probe_signal[i] = stack[:,hs[i],ws[i]]
    # Denoise
    sigmas[i] = estimate_sigma(probe_signal[i])
sigma = 1*np.max(sigmas)

# Now denoise
plt.rcParams["figure.figsize"] = (15,10)
figure, ax = plt.subplots(nrows=Nh*Nw, ncols=2)
for i in range(Nh*Nw):   
    probe_denoised = denoise_wavelet(probe_signal[i], sigma=sigma, **denoise_kwargs)
    ax[i,0].plot(probe_signal[i])
    ax[i,0].plot(probe_denoised)
    ax[i,1].plot(probe_signal[i][:60])
    ax[i,1].plot(probe_denoised[:60])
plt.show()


# If we are happy with results on the probes, do this for the whole stack (this can take a while):

# In[12]:


stack_denoised = np.empty_like(stack)
#stack_denoised = stack.copy()

for i in tqdm(range(height)):
    for j in range(width):
        stack_denoised[:, i, j] = denoise_wavelet(stack[:, i, j], sigma=sigma, **denoise_kwargs)


# Let's inspect the result on one image at time `t`:

# In[13]:

global t
t = 60
plt.rcParams["figure.figsize"] = (15,10)
figure, ax = plt.subplots(ncols=6)
ax[0].imshow(stack[t], vmin=0, vmax=1)
ax[1].imshow(stack_denoised[t], vmin=0, vmax=1)
ax[2].imshow(stack[t, height//3:2*height//3, width//3:2*width//3], vmin=0, vmax=1)
ax[3].imshow(stack_denoised[t, height//3:2*height//3, width//3:2*width//3], vmin=0, vmax=1)
ax[4].imshow(stack[t, 4*height//9:5*height//9, 4*width//9:5*width//9], vmin=0, vmax=1)
ax[5].imshow(stack_denoised[t, 4*height//9:5*height//9, 4*width//9:5*width//9], vmin=0, vmax=1)
plt.show()


# ## Spatial filtering of the absorbance using wavelets
# 
# The absorbance field is direclty linked to concentration (either proportional, or the result of some smooth transformation), therefore this field should be smooth in the gel.
# 
# We start with defining the absorbance and 'repare' outliers.

# In[14]:


A = -np.log10(stack_denoised/stack_denoised[0])
A[np.isnan(A)] = 0.
A[A<0] = 0.


# We now create a mask that identifies the probe shadow. This is done in two steps:
# 
# 1. Identify the interior of the probe which may by semi-transparent.
# 2. Identify all other pixels where the back-light intensity is not sufficient.

# In[15]:


## Create mask
mask = stack_denoised[0].copy()

# Probe envelope
min_intensity = 0.3

for i, line in enumerate(mask):
    # For each line
    j_min, j_max = -1, -1
    for j, value in enumerate(line):
        if value < min_intensity:
            if j_min == -1:
                j_min = j
            j_max = j
    if j_min > -1:
        mask[i, j_min:j_max] = np.nan

# Minimum backlight intensity in general
min_intensity_gen = 0.7
mask[mask<min_intensity_gen] = np.nan
        
plt.imshow(mask)
plt.show()


# It will help denoising if the absorbance has some non-zero value inside the probe. We fill the probe line by line with an average of the `k` greatest values outside the probe shadow:

# In[16]:


k = 5
for i in tqdm(range(len(A))):
    for l, line in enumerate(A[i]):
        # Line by line
        largest = np.partition(-A[i,l][~np.isnan(mask[l])], k)[:k]
        value = -np.average(largest)
        A[i,l][np.isnan(mask[l])] = value


# Let's inspect the result:

# In[17]:


plt.imshow(A[80])
plt.show()


# Now we are ready to perform the spatial filtering. This is done with a standard wavelet denoising algorithm from the `skimage.restoration` module. Parameters are stored in the following dictionary:

# In[18]:


from skimage.restoration import denoise_wavelet, estimate_sigma

denoise_kwargs = {'wavelet': 'sym3', 
                  'mode': 'soft',
                  'method': 'VisuShrink'
                  }


# Let's try the denoising on one absorbance image at time `t` first.

# In[19]:

global denoised_img
t = 60
sigma_est = estimate_sigma(A[t])
denoised_img = denoise_wavelet(A[t], sigma= sigma_est, **denoise_kwargs)

plt.rcParams["figure.figsize"] = (15,10)
figure, ax = plt.subplots(ncols=6)
vmin = np.min(A[t])
vmax = np.max(A[t])
ax[0].imshow(A[t], vmin=vmin, vmax=vmax)
ax[1].imshow(denoised_img, vmin=vmin, vmax=vmax)
ax[2].imshow(A[t, height//3:2*height//3, width//3:2*width//3], vmin=vmin, vmax=vmax)
ax[3].imshow(denoised_img[height//3:2*height//3, width//3:2*width//3], vmin=vmin, vmax=vmax)
ax[4].imshow(A[t, 4*height//9:5*height//9, 4*width//9:5*width//9], vmin=vmin, vmax=vmax)
ax[5].imshow(denoised_img[4*height//9:5*height//9, 4*width//9:5*width//9], vmin=vmin, vmax=vmax)
plt.show()


# We can also inspect a single line:

# In[20]:





# Interactive element
print('start inspecting a single line')
# Function to update the plots based on slider values
def slice_update_plots(*args):
    global slice_value 
    slice_value = slider1.get()
    print(t)
    print(slice_value)
    
    #update plot 1
    ax1.clear()
    ax1.plot(A[t, slice_value], label='Raw')
    ax1.plot(denoised_img[slice_value], label='Denoised')
    ax1.legend()
    canvas1.draw()

    # Update slider value labels
    label1.config(text=f'Slider 1: {slider1.get()}')

# Initialize the main window
root = tk.Tk()
root.title("Tkinter Sliders with Plots")
# Create sliders and labels
frame = ttk.Frame(root)
frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

slider1 = tk.Scale(frame, from_=0, to=height-1, orient=tk.HORIZONTAL,resolution=1, command=slice_update_plots)
slider1.pack()
slider1.set(height//2)
label1 = tk.Label(frame, text=f'Slider 1: {slider1.get()}')
label1.pack()

# Create the plot area
plot_frame = ttk.Frame(root)
plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

fig = Figure(figsize=(8, 6))

# Plot 1
ax1 = fig.add_subplot(111)
canvas1 = FigureCanvasTkAgg(fig, master=plot_frame)
canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Initialize the plots with default values
slice_update_plots()

# Start the Tkinter main loop
root.mainloop()


print(slice_value)

########end of GUI pop-up###########



# In[21]:
# If we are happy, let's do it for the whole stack:

A_denoised = np.empty_like(A)
#A_denoised = A.copy()

for i, im in tqdm(enumerate(A), total=len(list_files)):
    sigma_est = estimate_sigma(im)
    A_denoised[i] = denoise_wavelet(im, sigma= sigma_est, **denoise_kwargs)


# ## Write results to a file
# 
# We store the denoised stack, denoised absorbance and the shadow mask in a compressed file:

# In[22]:


np.save('absorbance', A_denoised)
np.save('mask', mask)

print(f'saved denoised absorbance and shadow mask' )

# Uncomment this line to save denoised image stack as well:
#np.save('stack', stack_denoised)
#print(f'saved stack_denoised' )
