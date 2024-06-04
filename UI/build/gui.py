
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer


from pathlib import Path

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"/Users/zd/Repos/ISMM_EPSRC_brain/UI/build/assets/frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

import capture

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
fpm=10
num_frames=10
n_oversampling=10
def initiate_handler():
    global fpm,num_frames,n_oversampling
    fpm = int(entry_2.get())
    num_frames = int(entry_4.get())
    n_oversampling = int(entry_5.get())
    capture.opencamera
    

# Exposure handler
exp_value=5000
def exposure_handler():
    global exp_value
    exp_value = int(entry_3.get())
    print('exposure is set to %i' % exp_value)
    capture.adjust(exp_value)



#button1
button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: print("button_1 clicked"),
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
    file=relative_to_assets("button_3.png"))
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
    highlightthickness=0
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
    highlightthickness=0
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
    highlightthickness=0
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
    highlightthickness=0
) # num of frames
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
    highlightthickness=0
)
entry_5.place( # n_oversample
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
