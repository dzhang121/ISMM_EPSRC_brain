import time

repeat_times = 1
images_per_rotation = 10
delay_repeat=5

def capture_and_save(repeat_number=0,frame_number=0):
    
    print(f'Repeat number {repeat_number} Frame number {frame_number}')

# Capture handler

def main():

    for repeat_number in range(repeat_times+1):
        for frame_number in range(images_per_rotation):
            capture_and_save(repeat_number,frame_number)
            #step to the next rotation

            time.sleep(0.1)
        time.sleep(delay_repeat)
        

    print(f'Capture complete')

if __name__ == "__main__":
    main()