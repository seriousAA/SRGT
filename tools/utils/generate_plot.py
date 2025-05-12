import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Load the data from the CSV file
file_path = '/home/liyuqiu/RS-PCT/gsd_data_after (copy).csv'

# Loading the data again
gsd_data = pd.read_csv(file_path)

# Filtering the data to include only values within (0, 1)
filtered_gsd_data = gsd_data[(gsd_data['GSD'] > 5) & (gsd_data['GSD'] <= 100)]

# Binning the GSD data within the range (0, 1)
bin_counts, bin_edges = np.histogram(filtered_gsd_data['GSD'], bins=20)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Normalize the y-values
bin_counts_normalized = (bin_counts-bin_counts.min()) / (bin_counts.max()-bin_counts.min())

# Create the figure and axis
fig, ax = plt.subplots()

# Increase the number of points for a smoother line
fine_x = np.linspace(bin_centers.min(), bin_centers.max(), num=400)
# Create a spline of order 3 (cubic spline) to interpolate the y-values
spline = interp1d(bin_centers, bin_counts_normalized+0.005, kind='quadratic')
fine_y = spline(fine_x)

# Plotting the line chart
ax.plot(fine_x, fine_y, color='#2c7bb6', linewidth=5)

# Plotting the bar chart
ax.bar(bin_centers, bin_counts_normalized, align='center', width=np.diff(bin_edges), color='#bcbcbc', edgecolor='black', linewidth=0.5)

# Adding arrows for axes
# Adding only horizontal grid lines
ax.yaxis.grid(True)
ax.xaxis.grid(False)

# Removing ticks, labels, legends, and x- and y-axis spines
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)
plt.tight_layout()
plt.savefig('/home/liyuqiu/RS-PCT/gsd_data_plot_2.jpg')
