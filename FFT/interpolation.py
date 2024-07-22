import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from datetime import datetime

# Load the Excel file
path = os.path.join(os.getcwd(), 'Desktop/new_folder', 'average_fft_values.xlsx')
data = pd.read_excel(path, sheet_name="Sheet1")  # Replace "Sheet1" with your actual sheet name

# Assuming your data columns are named "Elevation", "Average FFT Value", and "Date and Time"
elevations = data["Sudut ketinggian Matahari"]
fft_values = data["Nilai hasil FFT"]#["Average Hue Value"]#
times = pd.to_datetime(data["Date and Time"], format="%Y:%m:%d %H:%M:%S").dt.time

# Convert times to numeric values representing seconds since midnight for interpolation
times_numeric = times.map(lambda t: t.hour * 3600 + t.minute * 60 + t.second)

# Spline interpolation
spline = UnivariateSpline(elevations, fft_values, s=1.8)  # 10 or 1.8 fft s is a smoothing factor, adjust as needed

# Generate a smooth curve
elevation_smooth = np.linspace(elevations.min(), elevations.max(), 1000)
fft_smooth = spline(elevation_smooth)

# Calculate first derivative of the smoothed curve
first_derivative = np.gradient(fft_smooth, elevation_smooth)

# Find local maxima (where first derivative changes sign from positive to negative)
local_max_indices = np.where(np.diff(np.sign(first_derivative)) < 0)[0]

# Choose the first and second local maxima (adjust based on your data)
extreme_indices = local_max_indices[:4]

# Interpolate to find exact value at the extreme points
interp_elevations = elevation_smooth[extreme_indices]
interp_fft_values = fft_smooth[extreme_indices]

# Find corresponding times for extreme points in numeric format
extreme_times_numeric = np.interp(interp_elevations, elevations, times_numeric)

# Convert the interpolated numeric times back to time format (seconds to HH:MM:SS)
extreme_times = [datetime.utcfromtimestamp(t).strftime('%H:%M:%S') for t in extreme_times_numeric]

# Plot the data
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.scatter(elevations, fft_values, label="Nilai hasil FFT")
ax1.plot(elevation_smooth, fft_smooth, label="Interpolasi", color='blue')
ax1.scatter(interp_elevations, interp_fft_values, color='red', label="Extreme Point")

# Annotate the extreme points
for i, (elev, fft_val, time) in enumerate(zip(interp_elevations, interp_fft_values, extreme_times)):
    ax1.annotate(f"(Elv: {elev:.2f}, \nTime: {time})", 
                 (elev, fft_val), 
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center', 
                 fontsize=9, 
                 color='black')

# Customize plot
ax1.set_xlabel("Sudut Ketinggian Matahari", fontsize=12)
ax1.set_ylabel("Nilai hasil FFT", fontsize=12)
ax1.set_title("Elevation vs. FFT Value")
ax1.legend()
ax1.grid(True)  # Enable grid on the primary axis

# Add secondary x-axis at the top
ax2 = ax1.secondary_xaxis('top')
ax2.set_xlabel('Waktu', fontsize=12)

# Set ticks and labels for the secondary x-axis
num_ticks = len(ax1.get_xticks())
interval = max(len(times) // num_ticks, 1)
selected_labels = [times[i] for i in range(0, len(times), interval)]

# Make sure the number of ticks and labels match
if len(selected_labels) > num_ticks:
    selected_labels = selected_labels[:num_ticks]

ax2.set_xticks(ax1.get_xticks())
ax2.set_xticklabels([label.strftime('%H:%M:%S') for label in selected_labels], rotation=45, ha='right')

# Enable grid on the secondary axis
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()

# Print details of the extreme points
for i, (elev, fft_val, time) in enumerate(zip(interp_elevations, interp_fft_values, extreme_times)):
    print(f"Extreme Point {i+1} (Elevation): {elev}")
    print(f"Extreme Point {i+1} (FFT Value): {fft_val}")
    print(f"Extreme Point {i+1} (Time): {time}")
