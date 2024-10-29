import matplotlib.pyplot as plt
from collections import Counter
from termcolor import colored
import itertools

def calculate_frequencies(input_string, plot="print"):
    # Calculate the frequency of each character in the input string
    freq = Counter(input_string)
    if plot == "plot":
        plot_frequencies(freq)
    elif plot == "print":
        print_frequencies(freq)
    
    
    
def print_frequencies(frequencies):
    # Sort the frequencies dictionary by key (label) in ascending order
    sorted_frequencies = dict(sorted(frequencies.items(), key=lambda item: item[0]))
    
    labels = list(sorted_frequencies.keys())
    data = list(sorted_frequencies.values())
    
    # Determine the maximum frequency for scaling
    max_freq = max(data)
    scale = 50 / max_freq  # Scale to fit within 50 characters width

    # Define a list of colors to cycle through
    colors = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']
    color_cycle = itertools.cycle(colors)

    # Create a bar chart using ASCII characters
    for label, freq in zip(labels, data):
        color = next(color_cycle)
        bar = colored('█' * int(freq * scale), color)
        print(f"{label}: {bar} ({freq})")

    

def plot_frequencies(frequencies):
    # Plot the frequencies as a bar chart
    characters = list(frequencies.keys())
    counts = list(frequencies.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(characters, counts, color='blue')
    plt.xlabel('Characters')
    plt.ylabel('Frequency')
    plt.title('Character Frequency')
    plt.show()



  
  

