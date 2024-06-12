import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Initialize the main window
root = tk.Tk()
root.title("Tkinter Sliders with Plots")
# Create sliders and labels
frame = ttk.Frame(root)
frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

slider1 = tk.Scale(frame, from_=0, to=100, orient=tk.HORIZONTAL,resolution=1)
slider1.pack()
slider1.set(50)
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



# Start the Tkinter main loop
root.mainloop()

