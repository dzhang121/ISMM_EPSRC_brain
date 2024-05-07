import numpy as np
from skimage import io
#from skimage import filters
#from skimage import restoration
from skimage.util import img_as_float


import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.patches as patches


# Debug boolean. If True, print lots of things in terminal.
debug = True

# Display all. If True, each image and curve fitting is displayed. If false, 
# display first and last. User must close the window for the next image to process.
display_all = False

## Length calibration
# pixels per cm
pxpcm = 367.

## Convert pixel value to concentration (Beer-Lambert law)
# Molar extinction coefficient (cm^-1 L/mol)
epsilon = 74028.
# Methylene blue molar mass (g mol^-1)
M = 319.85

# Reference path
reference_filename = 'images/Basler_acA1920-150um__40076601__20210629_140629994_1.tiff'

# img_as_float converts unsinged integers data from floats within range [0,1]
ref0 = img_as_float(io.imread(reference_filename))

# Crop
#Middle
top_left = [950, 400]
bottom_right = [1270, 410]


ref = ref0[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]

rect = patches.Rectangle(top_left, bottom_right[0]-top_left[0], bottom_right[1]-top_left[1], linewidth=1, edgecolor='r', facecolor='none')

figure, ax = plt.subplots(ncols=2, constrained_layout=True)

ax[0].imshow(ref0, cmap='gray')
ax[0].add_patch(rect)
ax[0].set_title('Reference image')

ax[1].imshow(ref, cmap='gray')
ax[1].set_title('Extracted image')
plt.show()

## Prepare loop over images
time_min = 5
time_max = 65

# Array to record D
record = np.zeros((time_max-time_min,time_max-time_min))

for i,A in enumerate(range(time_min,time_max+1)):
    for j,B in enumerate(range(A+1,time_max+1)):
        if debug: print('i={}, j={}'.format(i,j))
        # Load images
        imA_filename = 'images/Basler_acA1920-150um__40076601__20210629_140629994_{}.tiff'.format(A)
        imB_filename = 'images/Basler_acA1920-150um__40076601__20210629_140629994_{}.tiff'.format(B)

        # dt in seconds
        time = (B-A)*60.

        imA0 = img_as_float(io.imread(imA_filename))
        imB0 = img_as_float(io.imread(imB_filename))
        absorbanceA = -np.log10(imA0[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]/ref)
        absorbanceB = -np.log10(imB0[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]/ref)
        absorbanceA[absorbanceA<0]=0
        absorbanceB[absorbanceB<0]=0
        if display_all:
            figure, ax = plt.subplots(ncols=2, constrained_layout=True)
            ax[0].imshow(absorbanceA, cmap='gray')
            ax[0].set_title('Initial image')

            ax[1].imshow(absorbanceB, cmap='gray')
            ax[1].set_title('After diffusion')
            plt.show()

        ## Averaging along lines
        absorbanceAm = np.average(absorbanceA, axis=0)
        absorbanceBm = np.average(absorbanceB, axis=0)

        # X axis
        X = np.arange(len(absorbanceAm))/pxpcm
        
        if display_all:
            plt.plot(X, absorbanceAm, label='Absorbance at {} min'.format(A))
            plt.plot(X, absorbanceBm, label='Absorbance at {} min'.format(B))



        # Fit image A
        def Gaussian(x, sigma, mu, a):
            return a*np.exp(-(x-mu)**2/(2*sigma**2))
        [sigmaA, muA, aA], pcov = curve_fit(Gaussian, X, absorbanceAm, p0=[0.1,0.5,1.])

        # Fit image B
        [sigmaB, muB, aB], pcov = curve_fit(Gaussian, X, absorbanceBm, p0=[0.1,0.5,1.])
        if display_all or i+j==0:
            plt.plot(X, Gaussian(X, sigmaA, muA, aA), label='Gaussian fit')
            plt.plot(X, Gaussian(X, sigmaB, muB, aB), label='Gaussian fit')
            plt.legend()
            plt.xlabel('Distance (cm)')
            plt.ylabel('Absorbance')
            plt.show()

        D = (sigmaB**2-sigmaA**2)/(2*time)
        if debug: print(D)
        record[i,i+j] = D
record[record==0] = np.nan
plt.imshow(record*1e6)
plt.ylabel('Time A (min)')
plt.xlabel('Time B (min)')
plt.colorbar(label='D ($10^{-6}$ cm$^2$/s)')
plt.show()

plt.hist(record.flatten(), bins=100, density=True)
plt.show()

print(f"Average diffusivity: {np.average(record[record>0])} cm²/s")
print(f"Median diffusivity: {np.median(record[record>0])} cm²/s")


