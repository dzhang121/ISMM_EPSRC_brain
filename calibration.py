import os
from natsort import natsorted, ns


import numpy as np
import scipy.ndimage
from skimage import io
from skimage.util import img_as_float
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erfc

import matplotlib.patches as patches

# Debug boolean. If True, print lots of things in terminal.
debug = True

# Display all. If True, each image in the stack is displayed. If false, 
# display first and last. User must close the window for the next image to process.
display_all = False

##
## Import images paths
##
# Concentrations in µg/ml
concentrations = np.array([
    0.0078125,
    0.015625,
    0.03125,
    0.0625,
    0.125,
    0.25,
    0.5,
    1.,
    2.,
    3.125,
    6.25,
    12.5,
    25.,
    50.,
    100.])
    
# Corresponding files
files = [
    'mb-0.0078125ugml.tiff',
    'mb-0.015625ugml.tiff',
    'mb-0.03125ugml.tiff',
    'mb-0.0625ugml.tiff',
    'mb-0.125ugml.tiff',
    'mb-0.25ugml.tiff',
    'mb-0.5ugml.tiff',
    'mb-1ugml.tiff',
    'mb-2ugml.tiff',
    'mb-3.125ugml.tiff',
    'mb-6.25ugml.tiff',
    'mb-12.5ugml.tiff',
    'mb-25ugml.tiff',
    'mb-50ugml.tiff',
    'mb-100ugml.tiff']

# Reference image file
ref_file = 'cuvette-water.tiff'

## Read the background/reference image

# Reference: first image of the stack
# img_as_float converts unsinged integers data from floats within range [0,1]
ref = img_as_float(io.imread(ref_file))

# Rectangle to extract
# Coordinates are numpy matrices, with origin at top left corner of the image
# (line:y, column:x).
# Large
top_left = [727, 25]
bottom_right = [1080, 1150]

# Small
#top_left = [900, 350]
#bottom_right = [1050, 500]

rect = patches.Rectangle(top_left, bottom_right[0]-top_left[0], bottom_right[1]-top_left[1], linewidth=1, edgecolor='r', facecolor='none')

figure, ax = plt.subplots(nrows=1, constrained_layout=True)

ax.imshow(ref, cmap='gray')
ax.add_patch(rect)
ax.set_title('Reference image')
plt.show()

# Prepare result logging
table_A = np.zeros(len(files))
table_var_A = np.zeros(len(files))

for i, f in enumerate(files):
    img = img_as_float(io.imread(f))
    
    absorbance = -np.log10(img[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]/ref[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]])
    
    A = np.average(absorbance)
    var_A = np.var(absorbance)
    
    table_A[i] = A
    table_var_A[i] = var_A
    if display_all or i==0:
    
        rect = patches.Rectangle(top_left, bottom_right[0]-top_left[0], bottom_right[1]-top_left[1], linewidth=1, edgecolor='r', facecolor='none')

        figure, ax = plt.subplots(ncols=3, constrained_layout=True)

        ax[0].imshow(img, cmap='gray')
        ax[0].add_patch(rect)
        ax[0].set_title(f)
        ax[1].imshow(absorbance, cmap='gray')
        ax[1].set_title('Absorbance')
        hist, bins, p = ax[2].hist(absorbance.flatten(), bins=200, density=True, label='Data')
        def normal(x, mu, var):
            return np.exp(-0.5*(x-mu)**2/var)/np.sqrt(2*np.pi*var)
        ax[2].plot(bins, normal(bins, A, var_A), label='Normal distribution')
        ax[2].legend()
        ax[2].set_title('Absorbance')
        
        plt.show()


# Theoretical absorbance
epsilon = 74028 # cm^-1·L/mol
ell = 1.        # cm
M = 319.85      # g·mol^-1
A_th = epsilon*ell/M*np.array(concentrations)*1e-3



plt.loglog(concentrations, table_A+np.sqrt(table_var_A), 'silver', label='One sigma')
plt.loglog(concentrations, table_A-np.sqrt(table_var_A), 'silver')
plt.loglog(concentrations, table_A, 'o-', label='Mean')
plt.loglog(concentrations, A_th, '-', label='Theoretical')

num_bits = 10
A_max, A_min = np.log10(2**num_bits-2), np.log10((2**num_bits-2)/(2**num_bits-3))
plt.loglog(concentrations, A_max*np.ones_like(concentrations), 'gray', label='Theoretical range for '+str(num_bits)+'-bit sensor')
plt.loglog(concentrations, A_min*np.ones_like(concentrations), 'gray')
plt.legend()
plt.xlabel('Concentrations (µg/ml)')
plt.ylabel('Absorbance')
plt.grid(which='major', color='lightgray')
plt.grid(which='minor', color='whitesmoke')
plt.show()

# Correction factor for small absorbance.
plt.loglog(table_A, A_th/(table_A+np.sqrt(table_var_A)), 'silver', label='One sigma')
plt.loglog(table_A, A_th/(table_A-np.sqrt(table_var_A)), 'silver')
plt.loglog(table_A, A_th/table_A, 'o-', label='Mean')
plt.loglog(table_A, np.ones_like(table_A), 'gray', label='1:1')
plt.loglog(table_A, (table_A/0.5)**0.125+0.5*(table_A/1.1)**8, label='correction model')
plt.legend()
plt.xlabel('Measured absorbance')
plt.ylabel('Correction factor')
plt.grid(which='major', color='lightgray')
plt.grid(which='minor', color='whitesmoke')
plt.show()



