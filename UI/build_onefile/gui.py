
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer

import os
from pathlib import Path

# from tkinter import *
# Explicit imports to satisfy Flake8
import tkinter
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / "assets/frame0"


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


# UI
window = Tk()

window.geometry("1000x900")
window.configure(bg = "#FFFFFF")


canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 900,
    width = 1000,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    32.0,
    117.0,
    805.0,
    209.0,
    fill="#ECECEC",
    outline="")

canvas.create_rectangle(
    0.0,
    0.0,
    774.0,
    81.0,
    fill="#21BB9F",
    outline="")

canvas.create_text(
    55.0,
    23.0,
    anchor="nw",
    text="Brain-in-a-Box Measurement Rig Control Panel",
    fill="#000000",
    font=("Inter", 30 * -1)
)

# Initiation handler
import numpy as np
import sched, time
from datetime import date

from skimage.io import imsave
from skimage.util import img_as_uint
from skimage import exposure

import matplotlib.pyplot as plt
from matplotlib import cm
from ipywidgets import interactive, IntRangeSlider, IntSlider

from pypylon import pylon

fpm=10
num_frames=10
n_oversampling=10
def initiate_handler():
    global fpm,num_frames,n_oversampling,camera
    fpm = int(entry_2.get())
    num_frames = int(entry_4.get())
    n_oversampling = int(entry_5.get())
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

# Exposure handler
exp_value=5000
def exposure_handler():
    global exp_value
    exp_value = int(entry_3.get())
    print('exposure is set to %i' % exp_value)
    adjust(exp_value)

# Capture 
def capture_and_save(frame_number=0):
    
    print('running capture' )
    img = grap_pseudo16bit(n_oversampling)

    file_name = f'{folder}/{date.today().isoformat()}_{frame_number:04d}.tiff'
    imsave(file_name, img)
    
    print(f'Frame number {frame_number}')

# Capture handler
def capture_handler():
    global folder
    folder = str(entry_1.get())
    if not os.path.exists(OUTPUT_PATH / folder):
        os.makedirs(OUTPUT_PATH / folder)
    print(f'pictures path: {OUTPUT_PATH / folder}')

    s = sched.scheduler(time.time, time.sleep)
    for i in range(num_frames):
        s.enter(i*60./fpm, 1, capture_and_save, kwargs={'frame_number':i})
        
    s.run()

    print(f'Capture complete')


###########################UI bits###########################
#button1 - capture button
button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=capture_handler,
    relief="flat"
)
button_1.place(
    x=50.0,
    y=639.0,
    width=200.0,
    height=85.0
)

# exposure update button
button_image_4 = PhotoImage(
    file=relative_to_assets("button_4.png"))
button_4 = Button(
    image=button_image_4,
    borderwidth=0,
    highlightthickness=0,
    command=exposure_handler,
    relief="flat"
)
button_4.place(
    x=244.0,
    y=429.0,
    width=200.0,
    height=85.0
)
#######
# initiate button
button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=initiate_handler,
    relief="flat"
)
button_2.place(
    x=32.0,
    y=335.0,
    width=200.0,
    height=85.0
)

button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_3 clicked"),
    relief="flat"
)
button_3.place(
    x=46.0,
    y=760.0,
    width=273.0,
    height=85.0
)

#path entry
entry_image_1 = PhotoImage(
    file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(
    423.0,
    179.5,
    image=entry_image_1
)
entry_1 = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0,
    textvariable=tkinter.StringVar(value='images')
)
entry_1.place(
    x=50.0,
    y=163.0,
    width=746.0,
    height=31.0
)

canvas.create_text(
    46.0,
    130.0,
    anchor="nw",
    text="Output path:",
    fill="#000000",
    font=("Inter", 20 * -1)
)

canvas.create_rectangle(
    32.0,
    236.0,
    232.0,
    315.0,
    fill="#ECECEC",
    outline="")

# entry2 fpm
entry_image_2 = PhotoImage(
    file=relative_to_assets("entry_2.png"))
entry_bg_2 = canvas.create_image(
    136.5,
    292.5,
    image=entry_image_2
)
entry_2 = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0,
    textvariable=tkinter.IntVar(value=10)
)
entry_2.place( #fpm
    x=50.0,
    y=276.0,
    width=173.0,
    height=31.0
)

canvas.create_text(
    46.0,
    249.0,
    anchor="nw",
    text="Frame per minute:",
    fill="#000000",
    font=("Inter", 20 * -1)
)

canvas.create_rectangle(
    32.0,
    434.0,
    232.0,
    513.0,
    fill="#ECECEC",
    outline="")

# Exposure entry
entry_image_3 = PhotoImage(
    file=relative_to_assets("entry_3.png"))
entry_bg_3 = canvas.create_image(
    136.5,
    490.5,
    image=entry_image_3
)
entry_3 = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0,
    textvariable=tkinter.IntVar(value=7000)
)
entry_3.place(
    x=50.0,
    y=474.0,
    width=173.0,
    height=31.0
)



canvas.create_text(
    46.0,
    447.0,
    anchor="nw",
    text="Exposure:",
    fill="#000000",
    font=("Inter", 20 * -1)
)

canvas.create_rectangle(
    253.0,
    236.0,
    453.0,
    315.0,
    fill="#ECECEC",
    outline="")

# num of frames
entry_image_4 = PhotoImage(
    file=relative_to_assets("entry_4.png"))
entry_bg_4 = canvas.create_image(
    357.5,
    293.5,
    image=entry_image_4
)
entry_4 = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0,
    textvariable=tkinter.IntVar(value=10)
) 
entry_4.place(
    x=271.0,
    y=277.0,
    width=173.0,
    height=31.0
)

canvas.create_text(
    267.0,
    249.0,
    anchor="nw",
    text="Number of frames:",
    fill="#000000",
    font=("Inter", 20 * -1)
)

canvas.create_rectangle(
    474.0,
    236.0,
    674.0,
    315.0,
    fill="#ECECEC",
    outline="")

# n_oversample
entry_image_5 = PhotoImage(
    file=relative_to_assets("entry_5.png"))
entry_bg_5 = canvas.create_image(
    578.5,
    293.5,
    image=entry_image_5
)
entry_5 = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0,
    textvariable=tkinter.IntVar(value=6)
)
entry_5.place( 
    x=492.0,
    y=277.0,
    width=173.0,
    height=31.0
)

canvas.create_text(
    479.0,
    249.0,
    anchor="nw",
    text="Oversampling index:",
    fill="#000000",
    font=("Inter", 20 * -1)
)

image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    433.0,
    484.0,
    image=image_image_1
)
window.resizable(False, False)
window.mainloop()
