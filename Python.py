import pandas as pd
import matplotlib.pyplot as plt

# List of CSV files to plot
sorting_names = ["SingleThread", "MultiThread"]
colors = ['b', 'r', 'g']  # Different colors for each dataset
markers = ['o', 's', '^']  # Different markers for each dataset

plt.figure(figsize=(10, 6))

for sorting_name, color, marker in zip(sorting_names, colors, markers):
    file_name = sorting_name + ".csv"
    try:
        data = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found. Skipping.")
        continue

    # Extract data for graphing
    num_elements = data["Array Size"]
    execution_time = data["Time (seconds)"]

    # Plot the data
    plt.plot(num_elements, execution_time, marker=marker, linestyle='-', color=color, label=sorting_name)

# Customize the graph
plt.title("Sorting Performance Comparison")
plt.xlabel("Number of Elements")
plt.ylabel("Execution Time (seconds)")
plt.grid(True)
plt.legend()

# Save the graph as an image
graph_file = "SortingPerformanceComparison.png"
plt.savefig(graph_file)
print(f"Graph saved as '{graph_file}'.")

# Display the graph
plt.show()
