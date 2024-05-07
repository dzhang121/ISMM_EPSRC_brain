
import serial
import sys
import time

# p_abs (bar) = 1.013 + 25*(V-0.5)/4
# V = 5*n/1023
# p_abs = 1.013 + (5*n/1023 - 0.5)/4


if __name__ == '__main__':
    ser = serial.Serial('COM4',9600)

    log_file_name = time.strftime("%Y-%m-%d_%H%M%S") + '_pressure_log.txt'
    
    
    while True:
        # Update data by reading serial port
        
        newdata = ser.readline()
        p = float(newdata.strip())/1000

        # Write to terminal
        sys.stdout.write("\r %5.3f +/- 0.02 kPa" % p)
        sys.stdout.flush()

        # Write to log file
        f = open(log_file_name, "a")
        f.write(str(time.time())+'\t'+str(p)+'\n')
        f.close()

