import re
import matplotlib.pyplot as plt

# Step 1: Read the text document
path = 'Model/testing/variant-0/run2/'
filename = 'slurm-5723115-run2.out'  # Change this to your document's filename
specific_phrase = 'ep_rew_mean'  # Change this to the phrase you're looking for
numbers = []

with open(path + filename, 'r') as file:
    for line in file:
        # Step 2: Find rows with the specific phrase
        if specific_phrase in line:
            # Step 3: Extract the number from the found rows
            # Assuming the number is an integer or float written in the line
            found_numbers = re.findall(r'\d+\.?\d*', line)
            numbers.extend(found_numbers)  # Assuming you want all numbers on the line

# Convert found numbers to float (or int, depending on your data)
numbers = [float(num) for num in numbers]

# Step 5: Visualize the data
plt.figure(figsize=(10, 6))
plt.plot(numbers, linestyle='-')
plt.title('Visualization of Numbers Found')
plt.xlabel('Occurrence')
plt.ylabel('Number')
plt.grid(True)
plt.show()
