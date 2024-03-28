import re
import numpy as np
import matplotlib.pyplot as plt


def moving_average(x, w):
    return np.convolve(x,np.ones(w), 'valid') / w
# Step 1: Read the text document
# runs = [1,2,4,5,6,8]
# runs = [5,6]
runs = [21]
# runs = [9,10,11]
numbers = {}
for run in runs:
    path = f'Model/training/variant-0/run{run}/'
    filename = f'slurm-run{run}.out'  # Change this to your document's filename
    specific_phrase = 'ep_rew_mean'  # Change this to the phrase you're looking for
    # specific_phrase = 'actor_loss'  # Change this to the phrase you're looking for
    
    numbers[run] = []
    with open(path + filename, 'r') as file:
        for line in file:
            # Step 2: Find rows with the specific phrase
            if specific_phrase in line:
                # Step 3: Extract the number from the found rows
                # Assuming the number is an integer or float written in the line
                found_numbers = re.findall(r'\d+\.?\d*', line)
                numbers[run].extend(found_numbers)  # Assuming you want all numbers on the line

    # Convert found numbers to float (or int, depending on your data)
    numbers[run] = [float(num) for num in numbers[run]]

# Step 5: Visualize the data
plt.figure(figsize=(10, 6))
for run in runs:
    plt.plot(moving_average(numbers[run],1), linestyle='-',linewidth=2)
plt.title('Visualization of Numbers Found')
plt.legend([f'run{run}' for run in runs])
plt.xlabel('Occurrence')
plt.ylabel('Number')
plt.grid(True)
plt.show()
