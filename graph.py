import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the CSV file
df = pd.read_csv("benchmark_results.csv")

# Set up the 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
sc = ax.scatter(
    df['batch_size'],
    df['max_tokens'],
    df['tokens_per_second'],
    c=df['temperature'],
    cmap='viridis',
    s=60
)

# Axis labels
ax.set_xlabel('Batch Size')
ax.set_ylabel('Max Tokens')
ax.set_zlabel('Tokens per Second')
plt.title('GPU Inference Benchmark')

# Colorbar for temperature
cbar = plt.colorbar(sc, pad=0.1)
cbar.set_label('Temperature')

plt.show()
