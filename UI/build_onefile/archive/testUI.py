import tkinter as tk
from tkinter import ttk

# Function to update the label with the slider's value
def on_slider_move(event):
    value = slider_value.get()
    label.config(text=f'Slider value: {value}')

# Create a Tkinter window
root = tk.Tk()
root.title("Single Slider Example")

# Create an IntVar to store the slider value
slider_value = tk.IntVar(value=50)

# Create a label to display the slider value
label = ttk.Label(root, text=f"Slider value: {slider_value.get()}")
label.pack(pady=10)

# Create a single slider
slider = ttk.Scale(root, from_=0, to=100, orient="horizontal", variable=slider_value, command=on_slider_move)
slider.pack(fill='x', padx=10)

# Function to get the slider value from the main script
def get_slider_value():
    return slider_value.get()

# Function to periodically print the slider value
def print_slider_value():
    print(f'The slider value is: {get_slider_value()}')
    root.after(1000, print_slider_value)  # Repeat every 1000 milliseconds (1 second)

# Start the periodic printing
#print_slider_value()

# Run the Tkinter event loop
root.mainloop()

# Retrieve the final slider value after the Tkinter loop ends
final_value = get_slider_value()
print(f'The final slider value is: {final_value}')