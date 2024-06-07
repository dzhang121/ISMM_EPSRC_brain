import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


# Function to update the plots based on slider values
def update_plots(*args):
    global slider_values
    slider_values = [slider1.get(), slider2.get(), slider3.get(), slider4.get()]
    
    # Update plot1
    x = np.linspace(0, 10, 100)
    y1 = slider_values[0] * np.sin(slider_values[1] * x)
    ax1.clear()
    ax1.plot(x, y1)
    canvas1.draw()
    
    # Update plot2
    y2 = slider_values[2] * np.cos(slider_values[3] * x)
    ax2.clear()
    ax2.plot(x, y2)
    canvas2.draw()
    
    # Update slider value labels
    label1.config(text=f'Slider 1: {slider1.get()}')
    label2.config(text=f'Slider 2: {slider2.get()}')
    label3.config(text=f'Slider 3: {slider3.get()}')
    label4.config(text=f'Slider 4: {slider4.get()}')



# Initialize the main window
root = tk.Tk()
root.title("Tkinter Sliders with Plots")

# Create sliders and labels
frame = ttk.Frame(root)
frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

slider1 = tk.Scale(frame, from_=0, to=10, orient=tk.HORIZONTAL, command=update_plots)
slider1.pack()
label1 = tk.Label(frame, text=f'Slider 1: {slider1.get()}')
label1.pack()

slider2 = tk.Scale(frame, from_=0, to=10, orient=tk.HORIZONTAL, command=update_plots)
slider2.pack()
label2 = tk.Label(frame, text=f'Slider 2: {slider2.get()}')
label2.pack()

slider3 = tk.Scale(frame, from_=0, to=10, orient=tk.HORIZONTAL, command=update_plots)
slider3.pack()
label3 = tk.Label(frame, text=f'Slider 3: {slider3.get()}')
label3.pack()

slider4 = tk.Scale(frame, from_=0, to=10, orient=tk.HORIZONTAL, command=update_plots)
slider4.pack()
label4 = tk.Label(frame, text=f'Slider 4: {slider4.get()}')
label4.pack()

# Create export button
export_button = ttk.Button(frame, text="Export Slider Values", command=update_plots)
export_button.pack()

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
update_plots()

# Start the Tkinter main loop
root.mainloop()


print(slider_values)