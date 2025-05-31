import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the CSV file
try:
    df = pd.read_csv("benchmark_results.csv")
except FileNotFoundError:
    print("Error: 'benchmark_results.csv' not found. Ensure the file is in the correct directory.")
    exit()

# Check the data
print("First 5 rows of the CSV:")
print(df.head())
print("\nData types of columns:")
print(df.dtypes)

# Convert columns to numeric, handling errors by coercing to NaN
for col in ['batch_size', 'max_tokens', 'tokens_per_second', 'temperature']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Check for missing values after conversion
if df.isnull().any().any():
    print("\nWarning: Missing or invalid values detected in the data:")
    print(df.isnull().sum())
    print("These will be ignored in the plot.")

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

# Save the plot to a file
plt.savefig('gpu_benchmark_3d_plot.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'gpu_benchmark_3d_plot.png' in the current directory.")

# Close the figure to free memory
plt.close()