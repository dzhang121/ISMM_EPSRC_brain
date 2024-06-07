import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Function to update the plot based on the slider values
def update_plot():
    min_val = min_slider.get()
    max_val = max_slider.get()
    x = np.linspace(0, 10, 100)
    y = np.sin(x * (min_val + max_val) / 2)  # Example: sine wave based on the average of min and max values
    ax.clear()
    ax.plot(x, y)
    ax.set_title(f"Plot with slider range: {min_val} - {max_val}")
    canvas.draw()

# Function to handle minimum slider movement
def on_min_slider_move(event):
    min_val = min_slider.get()
    max_val = max_slider.get()
    if min_val > max_val:
        min_slider.set(max_val)
    update_label()
    update_plot()

# Function to handle maximum slider movement
def on_max_slider_move(event):
    min_val = min_slider.get()
    max_val = max_slider.get()
    if max_val < min_val:
        max_slider.set(min_val)
    update_label()
    update_plot()

# Function to update the label with the slider values
def update_label():
    min_val = min_slider.get()
    max_val = max_slider.get()
    label.config(text=f'Selected range: {min_val} - {max_val}')

# Create a Tkinter window
root = tk.Tk()
root.title("Range Slider and Plot Example")

# Create an IntVar to store the slider values
min_slider_value = tk.IntVar(value=20)
max_slider_value = tk.IntVar(value=80)

# Create a label to display the slider values
label = ttk.Label(root, text="Selected range: 20 - 80")
label.pack(pady=10)

# Create two sliders: one for the minimum value and one for the maximum value
min_slider = ttk.Scale(root, from_=0, to=100, orient="horizontal", variable=min_slider_value, command=on_min_slider_move)
max_slider = ttk.Scale(root, from_=0, to=100, orient="horizontal", variable=max_slider_value, command=on_max_slider_move)

min_slider.pack(fill='x', padx=10)
max_slider.pack(fill='x', padx=10)

# Create a matplotlib figure and axis
fig, ax = plt.subplots()

# Create a FigureCanvasTkAgg object
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill='both', expand=True)

# Initial plot update
update_plot()

# Run the Tkinter event loop
root.mainloop()

# Retrieve the final selected range after the Tkinter loop ends
final_min_val = min_slider.get()
final_max_val = max_slider.get()
print(f'The final selected range is: {final_min_val} - {final_max_val}')