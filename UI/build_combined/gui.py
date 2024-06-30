import os
from pathlib import Path

# from tkinter import *
# Explicit imports to satisfy Flake8
import tkinter
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage

from time import sleep

import subprocess

import numpy as np
import sched, time
from datetime import date

from skimage.io import imsave
from skimage.util import img_as_uint
from skimage import exposure

import matplotlib.pyplot as plt
from matplotlib import cm
from ipywidgets import interactive, IntRangeSlider, IntSlider

# from pypylon import pylon

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / "assets/frame0"


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

# In[] Handlers
fpm=10
images_per_rotation=360
num_frames=10
delay_repeat=30
repeat_times=0
n_oversampling=10

def initiate_handler():
    global fpm,num_frames,n_oversampling,camera,images_per_rotation,repeat_times,delay_repeat,angle_step
    images_per_rotation = int(entry_2.get())
    repeat_times = int(entry_4.get())
    delay_repeat  = int(entry_5.get())
    n_oversampling = int(entry_6.get())

    num_frames=images_per_rotation*repeat_times
    angle_step=360/images_per_rotation

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

# Update button - Exposure handler 
exp_value=5000
def exposure_handler():
    global exp_value
    exp_value = int(entry_3.get())
    print('exposure is set to %i' % exp_value)
    adjust(exp_value)


def capture_and_save(repeat_number=0,frame_number=0):
    
    print('running capture' )
    img = grap_pseudo16bit(n_oversampling)

    file_name = f'{folder}/{date.today().isoformat()}_{repeat_number:03d}_{frame_number:04d}.tiff'
    imsave(file_name, img)
    
    print(f'Repeat number {repeat_number} Frame number {frame_number}')


# 3D Capture handler
def capture_3D_handler():
    global folder
    folder = str(entry_1.get())
    if not os.path.exists(OUTPUT_PATH / folder):
        os.makedirs(OUTPUT_PATH / folder)
    print(f'pictures path: {OUTPUT_PATH / folder}')

    from servo_control import send_servo_position

    for repeat_number in range(repeat_times+1):
        angle = 0.0
        for frame_number in range(images_per_rotation):
            send_servo_position(angle)
            sleep(0.3)
            capture_and_save(repeat_number,frame_number)
            sleep(0.3)
            angle = angle+angle_step

        send_servo_position(0.0)

        sleep(delay_repeat)
        

    print(f'Capture complete')

# 2D Capture handler
def capture_2D_handler():
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


# Post-process Handler
def process_2D_handler():
    
    print(f'analysing images' )
    subprocess.run(['python','1 Preprocessing.py'],capture_output=False,check=True)
    #exec(open('1 Preprocessing.py').read())
    print(f'1.Preprocessing finished' )

    sleep(1.1)
    subprocess.run(['python','2 Integrate.py'],capture_output=False,check=True)
    print(f'2 Integrate finished' )

    sleep(1.1)
    subprocess.run(['python','3 Reconstruct.py'],capture_output=False,check=True)
    print(f'3 Reconstruct finished' )

    print(f'Process: all finished, images analysed' )


# Result Handler
def result_2D_handler():
    
    print(f'reading results' )
    subprocess.run(['python','4 Result.py'],capture_output=False,check=True)
    print(f'finished visualising results' )
    


# In[] UI elements
window = Tk()

window.geometry("900x990")
window.configure(bg = "#FFFFFF")


canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 990,
    width = 900,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    318.0,
    612.0,
    582.0,
    981.0,
    fill="#F8F5FF",
    outline="")

canvas.create_rectangle(
    606.0,
    608.0,
    870.0,
    981.0,
    fill="#FFF5EB",
    outline="")

canvas.create_rectangle(
    34.0,
    191.0,
    860.0,
    335.0,
    fill="#F8FAFF",
    outline="")

canvas.create_rectangle(
    37.0,
    612.0,
    302.0,
    981.0,
    fill="#F8FAFF",
    outline="")

canvas.create_rectangle(
    35.0,
    347.0,
    861.0,
    491.0,
    fill="#F8F5FF",
    outline="")

canvas.create_rectangle(
    30.0,
    79.0,
    860.0,
    171.0,
    fill="#ECECEC",
    outline="")

canvas.create_rectangle(
    0.0,
    0.0,
    900.0,
    64.0,
    fill="#21BB9F",
    outline="")

canvas.create_text(
    105.0,
    16.0,
    anchor="nw",
    text="Brain-in-a-Box Measurement Rig Control Panel",
    fill="#000000",
    font=("Inter", 30 * -1)
)

canvas.create_text(
    54.0,
    206.0,
    anchor="nw",
    text="2D Capture Parameters",
    fill="#000000",
    font=("Inter Bold", 20 * -1)
)

canvas.create_text(
    111.0,
    625.0,
    anchor="nw",
    text="2D Controls",
    fill="#000000",
    font=("Inter Bold", 20 * -1)
)

canvas.create_text(
    385.0,
    625.0,
    anchor="nw",
    text="3D Controls",
    fill="#000000",
    font=("Inter Bold", 20 * -1)
)

canvas.create_text(
    660.0,
    625.0,
    anchor="nw",
    text="Manual Override",
    fill="#000000",
    font=("Inter Bold", 20 * -1)
)

canvas.create_text(
    54.0,
    361.0,
    anchor="nw",
    text="3D Capture Parameters",
    fill="#000000",
    font=("Inter Bold", 20 * -1)
)
# In[] Process 2D
button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=process_2D_handler,
    relief="flat"
)
button_1.place(
    x=71.0,
    y=773.0,
    width=200.0,
    height=85.0
)
# In[] Capture 2D
button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=capture_2D_handler,
    relief="flat"
)
button_2.place(
    x=71.0,
    y=670.0,
    width=200.0,
    height=85.0
)
# In[] Capture 3D
button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=capture_3D_handler,
    relief="flat"
)
button_3.place(
    x=350.0,
    y=668.0,
    width=200.0,
    height=85.0
)
# In[] Rotate
button_image_4 = PhotoImage(
    file=relative_to_assets("button_4.png"))
button_4 = Button(
    image=button_image_4,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_4 clicked"),
    relief="flat"
)
button_4.place(
    x=638.0,
    y=773.0,
    width=200.0,
    height=85.0
)
# In[] Save photo
button_image_5 = PhotoImage(
    file=relative_to_assets("button_5.png"))
button_5 = Button(
    image=button_image_5,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_5 clicked"),
    relief="flat"
)
button_5.place(
    x=638.0,
    y=877.0,
    width=200.0,
    height=85.0
)
# In[] Result 2D
button_image_6 = PhotoImage(
    file=relative_to_assets("button_6.png"))
button_6 = Button(
    image=button_image_6,
    borderwidth=0,
    highlightthickness=0,
    command=result_2D_handler,
    relief="flat"
)
button_6.place(
    x=71.0,
    y=877.0,
    width=200.0,
    height=85.0
)
# In[] Initiate 
button_image_7 = PhotoImage(
    file=relative_to_assets("button_7.png"))
button_7 = Button(
    image=button_image_7,
    borderwidth=0,
    highlightthickness=0,
    command=initiate_handler,
    relief="flat"
)
button_7.place(
    x=71.0,
    y=511.0,
    width=200.0,
    height=85.0
)
# In[] Update
button_image_8 = PhotoImage(
    file=relative_to_assets("button_8.png"))
button_8 = Button(
    image=button_image_8,
    borderwidth=0,
    highlightthickness=0,
    command=exposure_handler,
    relief="flat"
)
button_8.place(
    x=638.0,
    y=511.0,
    width=200.0,
    height=85.0
)
# In[] output path entry
entry_image_1 = PhotoImage(
    file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(
    438.0,
    141.5,
    image=entry_image_1
)
entry_1 = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_1.place(
    x=48.0,
    y=125.0,
    width=780.0,
    height=31.0
)

canvas.create_text(
    44.0,
    92.0,
    anchor="nw",
    text="Output path:",
    fill="#000000",
    font=("Inter", 20 * -1)
)
# In[] frame per minute entry
canvas.create_rectangle(
    47.0,
    241.0,
    297.0,
    320.0,
    fill="#ECECEC",
    outline="")

entry_image_2 = PhotoImage(
    file=relative_to_assets("entry_2.png"))
entry_bg_2 = canvas.create_image(
    174.0,
    297.5,
    image=entry_image_2
)
entry_2 = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_2.place(
    x=64.0,
    y=281.0,
    width=220.0,
    height=31.0
)

canvas.create_text(
    61.0,
    254.0,
    anchor="nw",
    text="Frame per minute:",
    fill="#000000",
    font=("Inter", 20 * -1)
)
# In[] exposure entry
canvas.create_rectangle(
    322.0,
    511.0,
    572.0,
    590.0,
    fill="#ECECEC",
    outline="")

entry_image_3 = PhotoImage(
    file=relative_to_assets("entry_3.png"))
entry_bg_3 = canvas.create_image(
    449.0,
    566.5,
    image=entry_image_3
)
entry_3 = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_3.place(
    x=339.0,
    y=550.0,
    width=220.0,
    height=31.0
)

canvas.create_text(
    338.0,
    518.0,
    anchor="nw",
    text="Exposure:",
    fill="#000000",
    font=("Inter", 20 * -1)
)

# In[] rotation angle entry
canvas.create_rectangle(
    638.0,
    676.0,
    838.0,
    755.0,
    fill="#ECECEC",
    outline="")

entry_image_4 = PhotoImage(
    file=relative_to_assets("entry_4.png"))
entry_bg_4 = canvas.create_image(
    742.5,
    732.5,
    image=entry_image_4
)
entry_4 = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_4.place(
    x=656.0,
    y=716.0,
    width=173.0,
    height=31.0
)

canvas.create_text(
    649.0,
    683.0,
    anchor="nw",
    text="Rotation angle:",
    fill="#000000",
    font=("Inter", 20 * -1)
)

# In[] images per rotation entry
canvas.create_rectangle(
    47.0,
    398.0,
    297.0,
    477.0,
    fill="#ECECEC",
    outline="")

entry_image_5 = PhotoImage(
    file=relative_to_assets("entry_5.png"))
entry_bg_5 = canvas.create_image(
    173.0,
    454.5,
    image=entry_image_5
)
entry_5 = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_5.place(
    x=63.0,
    y=438.0,
    width=220.0,
    height=31.0
)

canvas.create_text(
    61.0,
    411.0,
    anchor="nw",
    text="Images per rotation:",
    fill="#000000",
    font=("Inter", 20 * -1)
)

# In[] number of repeats entry
canvas.create_rectangle(
    322.0,
    398.0,
    572.0,
    477.0,
    fill="#ECECEC",
    outline="")

entry_image_6 = PhotoImage(
    file=relative_to_assets("entry_6.png"))
entry_bg_6 = canvas.create_image(
    448.0,
    454.5,
    image=entry_image_6
)
entry_6 = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_6.place(
    x=338.0,
    y=438.0,
    width=220.0,
    height=31.0
)

canvas.create_text(
    336.0,
    411.0,
    anchor="nw",
    text="Number of repeats:",
    fill="#000000",
    font=("Inter", 20 * -1)
)

# In[] delay between repeats entry
canvas.create_rectangle(
    596.0,
    398.0,
    846.0,
    477.0,
    fill="#ECECEC",
    outline="")

entry_image_7 = PhotoImage(
    file=relative_to_assets("entry_7.png"))
entry_bg_7 = canvas.create_image(
    716.0,
    454.5,
    image=entry_image_7
)
entry_7 = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_7.place(
    x=606.0,
    y=438.0,
    width=220.0,
    height=31.0
)

canvas.create_text(
    609.0,
    409.0,
    anchor="nw",
    text="Delay between repeats:",
    fill="#000000",
    font=("Inter", 20 * -1)
)

# In[] number of frames entry
canvas.create_rectangle(
    322.0,
    241.0,
    572.0,
    320.0,
    fill="#ECECEC",
    outline="")

entry_image_8 = PhotoImage(
    file=relative_to_assets("entry_8.png"))
entry_bg_8 = canvas.create_image(
    446.0,
    298.5,
    image=entry_image_8
)
entry_8 = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_8.place(
    x=336.0,
    y=282.0,
    width=220.0,
    height=31.0
)

canvas.create_text(
    336.0,
    254.0,
    anchor="nw",
    text="Number of frames:",
    fill="#000000",
    font=("Inter", 20 * -1)
)

# In[] oversampling index entry
canvas.create_rectangle(
    597.0,
    241.0,
    847.0,
    320.0,
    fill="#ECECEC",
    outline="")

entry_image_9 = PhotoImage(
    file=relative_to_assets("entry_9.png"))
entry_bg_9 = canvas.create_image(
    721.0,
    298.5,
    image=entry_image_9
)
entry_9 = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_9.place(
    x=611.0,
    y=282.0,
    width=220.0,
    height=31.0
)

canvas.create_text(
    606.0,
    254.0,
    anchor="nw",
    text="Oversampling index:",
    fill="#000000",
    font=("Inter", 20 * -1)
)


# In[] 
window.resizable(False, False)
window.mainloop()
