import serial
import time

def send_servo_position(position, arduino_port='/dev/cu.usbmodem11301', baud_rate=9600):
    """
    Sends a servo position to the Arduino.

    Parameters:
    position (float): The position to set the servo to (0.0 - 360.0).
    arduino_port (str): The port where the Arduino is connected.
    baud_rate (int): The baud rate for serial communication.
    """
    try:
        position = float(position)
        if 0.0 <= position <= 360.0:
            with serial.Serial(arduino_port, baud_rate, timeout=1) as ser:
                ser.write(f"{position}\n".encode('utf-8'))  # Send the position as a string with a newline
                time.sleep(1)  # Wait for the Arduino to process the command
                response = ser.readline().decode('utf-8').strip()
                print(f"Arduino response: {response}")
        else:
            print("Error: Please enter a value between 0.0 and 180.0.")
    except ValueError:
        print("Error: Invalid input. Please enter a number.")


def send_servo_microsecond(position, arduino_port='/dev/cu.usbmodem11301', baud_rate=9600):
# this works with the corresponding firmware only
    try:
        position = float(position)
        with serial.Serial(arduino_port, baud_rate, timeout=1) as ser:
            ser.write(f"{position}\n".encode('utf-8'))  # Send the position as a string with a newline
            time.sleep(1)  # Wait for the Arduino to process the command
            response = ser.readline().decode('utf-8').strip()
            print(f"Arduino response: {response}")
        
    except ValueError:
        print("Error")

        