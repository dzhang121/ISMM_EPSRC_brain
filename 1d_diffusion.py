# The following modules are imported:
import os
from natsort import natsorted, ns
import numpy as np
import scipy.ndimage
from skimage import io
from skimage.util import img_as_float
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erfc

# Debug boolean. If True, print lots of things in terminal.
debug = True

# Display all. If True, each image and curve fitting is displayed. If false, display first and last. User must close the window for the next image to process.
display_all = False

## Length calibration
# pixels per cm
pxpcm = 367.

## Convert pixel value to concentration (Beer-Lambert law)
# Molar extinction coefficient (cm^-1 L/mol)
epsilon = 74028.
# Path lenth (cm)
ell = 1.
# Methylene blue molar mass (g mol^-1)
M = 319.85

## Import
# Folder containing the images. This should be the relative path from the python script.
folder = 'images'

# Get images names. 'natsorted' function will sort the names by 'natural' order, for example if there is a suffix with image number or timestamp.
list_files = natsorted(os.listdir(folder))
if debug: 
    for filename in list_files: print(filename)

## Read the background/reference image
# Reference: first image of the stack
# img_as_float converts unsinged integers data from floats within range [0,1]
ref = img_as_float(io.imread(folder+'/'+list_files[0]))

# Line to extract values from. Coordinates are numpy matrices, with origin at
# top left corner (line:y, column:x).
start_point = [265, 1120]
end_point = [1130, 1120]
# Distance in pixels from start_point to the air interface. This is used in the erfc function.
shift = 10

# Number of points along line
num_points = 1000
line = np.linspace(start_point, end_point, num_points)

# Length vector (in pixels)
s = np.sqrt((line[:,0] - start_point[0])**2+(line[:,1] - start_point[1])**2)

## Show reference image with extraction line
plt.imshow(ref, cmap='gray')
plt.plot([start_point[1], end_point[1]], [start_point[0], end_point[0]], 'ro-')
plt.title('Reference image and extraction line')
plt.show()

## Prepare for loop
# step size: if 1, read all images, if 10, read every other 10 images, etc.
step_i = 60
num_images = (len(list_files)-1)//step_i
if debug: print("Number of images to analyse: "+str(num_images))

# Table of fitted parameters (third row is used for error residual)
fit_results = np.zeros((num_images,3))

## Loop over images in the stack
for i in range(num_images):
    if debug: print('Image #i = '+str(i))

    # Load image as matrix of floats, values in [0,1]
    im = img_as_float(io.imread(folder+'/'+list_files[step_i*(1+i)]))

    # Ratiometric output. 1e-6 prevents division by zero.
    ratio = im/(ref+1e-6)
    # Threshold image values. Values higher than 1 mean reference pixel is more
    # opaque than current pixel
    ratio[ratio>1.] = 1.

    # Extract data along line
    extract = scipy.ndimage.map_coordinates(ratio, np.transpose(line))

    # Absorbance value
    absorbance = -np.log10(extract)
    
    # Concentration in ug/ml, as given by Beer-Lambert
    concentration = M*1e3/ell/epsilon*absorbance

    if display_all or i==0 or i==num_images-1:
        fig, axes = plt.subplots(nrows=2, constrained_layout=True)
        axes[0].set_title('...'+list_files[step_i*(1+i)][-15:-1]+': absorbance')
        im=axes[0].imshow(-np.log10(ratio), cmap='gray')
        fig.colorbar(im, ax=axes[0])
        axes[0].plot([start_point[1], end_point[1]], [start_point[0], end_point[0]], 'ro-')

    ## Fit erfc with floating coefficients
    def func(x, a, b):
        return a * erfc((x+shift)/b)
   
    popt, pcov = curve_fit(func, s, concentration, p0=[5,100])
    if debug: print('Fit result: '+str(popt)+', sigma = '+str(pcov[1,1]))
    # Save the fit results
    fit_results[i] = np.hstack((popt, np.sqrt(pcov[1,1])))
    
    if display_all or i==0 or i==num_images-1:
        axes[1].set_title('Beer-Lambert law') 
        axes[1].plot(s/pxpcm, concentration, label='data')
        axes[1].plot(s/pxpcm, func(s, popt[0], popt[1]), label='erfc fit')
        axes[1].set_xlabel('distance (cm)')
        axes[1].set_ylabel('Concentration (ug/ml)')
        axes[1].legend()
        plt.show()

## Time analysis

# Time line (in minutes)
time = np.linspace(1,(len(list_files)-1),num_images)

# Fit linear diffusion, D in pixel^2/min
[D], pcov = curve_fit(lambda x,D: np.sqrt(4*D*x), time[mask], fit_results[mask,1])

# Conversion, D0 in cm^2/s
D0 = D*(1./pxpcm)**2/60

plt.loglog(time, fit_results[:,1]/pxpcm, 'o-', label='data')
plt.loglog(time, np.sqrt(4*D*time)/pxpcm, label="sqrt fit: D={:.2e} cm^2/s".format(D0))
plt.legend()
plt.grid(which='major')
plt.grid(which='minor')
plt.xlabel('Time (min)')
plt.ylabel('Diffusion scale (cm)')
plt.show()

## Record in text file
np.savetxt('time_scale.txt', np.stack((time,fit_results[:,1]/pxpcm), axis=-1)
, header='(min) (cm)')
