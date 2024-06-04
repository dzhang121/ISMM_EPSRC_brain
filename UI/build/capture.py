import numpy as np
import sched, time
from datetime import date

from skimage.io import imsave
from skimage.util import img_as_uint
from skimage import exposure

#matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import cm
from ipywidgets import interactive, IntRangeSlider, IntSlider

from pypylon import pylon


# Folder (relative path from this script)
folder = 'images'

# Frame par minute
#fpm = 10

# Number of frames to record
#num_frames = 10

# Oversampling index (will average over 2**n_oversampling shots)
# Recommanded value: 6 (will sum 64 images and the result in a 16-bit array)
# Recommanded maximum: 12 (will average 4096 images, resulting in pseudo 16-bit image)
#n_oversampling = 6

def opencamera():
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()

def grap_pseudo16bit(n_oversampling):
    """Sum 2**n_oversampling images from the 10-bit camera and rescqle to obtain one pseudo-16-bit image."""
    img32 = np.zeros((camera.Height.GetValue(), camera.Width.GetValue()), dtype=np.uint32)
    
    # Take images
    camera.StartGrabbingMax(2**n_oversampling)
    
    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            # Access the image data.
            img32 += grabResult.Array

        grabResult.Release()
    # Rescale to uint16
    if n_oversampling < 6:
        # multiply by 2**(6-n_oversamp)
        img16 = np.left_shift(img32, 6-n_oversampling).astype(np.uint16)
    elif n_oversampling > 6:
        # devide by 2**(n_oversamp-6)
        img16 = np.right_shift(img32, n_oversampling-6).astype(np.uint16)
    else:
        # Just cast to 16 bits
        img16 = img32.astype(np.uint16)
    
    return img16

#adjust settings
# Initial value
#init_value = int(camera.ExposureTime.GetValue())
# Emulation
#init_value = 10_000

def adjust(exposure):
    # Set new exposure
    camera.ExposureTime.SetValue(float(exposure))
    
    # Get image
    img = grap_pseudo16bit(n_oversampling)
    
    # Emulation
    #img = np.random.randint(0,1023,size=(1000,1000), dtype=np.uint16)
    
    plt.rcParams["figure.figsize"] = (15,15)
    cmap = cm.get_cmap("gray").copy()
    cmap.set_over('red')
    
    figure, ax = plt.subplots(nrows=2)
    
    ax[0].imshow(img, cmap=cmap, vmin=0, vmax=2**16-2**6-1)
    
    ax[1].hist(img.flatten(), bins=256, log=True)
    plt.show()
    
    result = {'exposure': exposure,
             'width': 0,
             'height': 0}
    
    return result
    
#w = interactive(adjust, exposure=IntSlider(value=init_value,min=5000,max=12000, continuous_update=False))
#w

def capturef(frame_number=0):
    img = grap_pseudo16bit(n_oversampling)
    file_name = f'{folder}/{date.today().isoformat()}_{frame_number:04d}.tiff'
    imsave(file_name, img)
    
    print(f'Frame number {frame_number}')

    
    
# Schedule captures and run immediately

def start_capture_schedule():
    s = sched.scheduler(time.time, time.sleep)

    for i in range(num_frames):
        s.enter(i*60./fpm, 1, capturef, kwargs={'frame_number':i})
        
    s.run()