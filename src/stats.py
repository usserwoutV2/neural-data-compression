import matplotlib.pyplot as plt
from collections import Counter

def calculate_frequencies(input_string):
    # Calculate the frequency of each character in the input string
    plot_frequencies( Counter(input_string))

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



  
  

