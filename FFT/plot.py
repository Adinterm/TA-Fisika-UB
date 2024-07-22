
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def smooth_data(data, window_size=5):
    """Smooth the data using a simple moving average."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def extract_time(df, datetime_column):
    """Extract the time part from the datetime column and add it as a new column."""
    df['Time'] = df[datetime_column].apply(lambda x: x.split()[1] if isinstance(x, str) else '')
    return df

def plot_excel_data(file_path, column_names, x_column, secondary_x_column=None, use_dots=True, smooth=False):
    # Read the Excel file
    df = pd.read_excel(file_path)

    # Extract the time part from the 'Date and Time' column if present
    #if 'Date and Time' in df.columns:
    #    df = extract_time(df, 'Date and Time')

    # Create a scaler for normalization
    scaler = MinMaxScaler()

    # Create x-axis data
    if x_column and x_column in df.columns:
        x_data = df[x_column].values
    else:
        x_data = range(len(df))

    # Create secondary x-axis data
    if secondary_x_column and secondary_x_column in df.columns:
        secondary_x_data = df[secondary_x_column].values
    else:
        secondary_x_data = None

    # Plot for each specified column
    fig, ax1 = plt.subplots(figsize=(10, 5))

    for column_name in column_names:
        # Check if the specified column exists
        if column_name not in df.columns:
            print(f"Column '{column_name}' does not exist in the Excel file.")
            continue

        # Extract the data for the specified column
        data = df[column_name].values

        # Normalize the data
        data_normalized = scaler.fit_transform(data.reshape(-1, 1)).flatten()

        # Smooth the data if requested
        if smooth:
            data_normalized = smooth_data(data_normalized)

            # Adjust x_data for the smoothed data
            x_data = x_data[len(x_data) - len(data_normalized):]
            if secondary_x_data is not None:
                secondary_x_data = secondary_x_data[len(secondary_x_data) - len(data_normalized):]

        # Plot the data with or without dots
        if use_dots:
            ax1.plot(x_data, data_normalized, linestyle='-', marker='o', alpha=0.7, label=column_name)
        else:
            ax1.plot(x_data, data_normalized, linestyle='-', alpha=0.7, label=column_name)

    ax1.set_title('Analisa Perubahan Langit saat Fajar')
    ax1.set_xlabel(x_column if x_column else 'Total Data Points', fontsize=12)
    ax1.set_ylabel('Nilai Normalisasi', fontsize=12)
    ax1.grid()
    ax1.legend()

    if secondary_x_data is not None:
        ax2 = ax1.secondary_xaxis('top')
        ax2.set_xlabel(secondary_x_column, fontsize=12)
        num_ticks = len(ax1.get_xticks())
        interval = len(secondary_x_data) // num_ticks
        selected_labels = [secondary_x_data[i * interval] for i in range(num_ticks)]
        ax2.set_xticks(ax1.get_xticks())
        ax2.set_xticklabels(selected_labels)

    plt.show()

# Example usage
path = os.path.join(os.getcwd(), 'Desktop/new_folder', 'average_fft_values.xlsx')
file_path = path  # Replace with your Excel file path
column_names = ['Nilai hasil FFT', 'Nilai hasil HSV', 'Nilai hasil RGB']  # Replace with the columns you want to plot
x_column = 'Sudut ketinggian Matahari'  # Replace with the column you want to use for the x-axis (optional)
secondary_x_column = 'Waktu'  # Replace with the secondary x-axis column you want to use (optional)
plot_excel_data(file_path, column_names, x_column=x_column, secondary_x_column=secondary_x_column, use_dots=True, smooth=True)


import sys
sys.exit()

import pandas as pd
import os

path = os.path.join(os.getcwd(), 'Desktop/new_folder', 'average_fft_values.xlsx')
file_path = path  # Replace with your Excel file path

def extract_time_and_save(file_path):
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Check if the 'Date and Time' column exists
    if 'Date and Time' in df.columns:
        # Extract the time portion from the 'Date and Time' column
        df['Time'] = df['Date and Time'].apply(lambda x: x.split()[1] if isinstance(x, str) else '')
    
    # Define the new file path
    output_file_path = file_path.replace('.xlsx', '_with_time.xlsx')
    
    # Save the updated DataFrame to a new Excel file
    df.to_excel(output_file_path, index=False)
    print(f"Updated file saved as {output_file_path}")

# Example usage
file_path = file_path #'path_to_your_file.xlsx'  # Replace with your Excel file path
extract_time_and_save(file_path)



import sys
sys.exit()


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
import sys
sys.exit()
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

folder_name = "Desktop/images"
folder_path = os.path.join(os.getcwd(), folder_name)

# Print the folder path for debugging
print(f"Looking for images in: {folder_path}")

# Check if the folder exists
if not os.path.exists(folder_path):
    print(f"Folder does not exist: {folder_path}")
else:
    print(f"Folder found: {folder_path}")

# List the contents of the folder
print("Contents of the folder:")
print(os.listdir(folder_path))

# Define the threshold for the high-pass filter
high_pass_threshold = 20  # Adjust this value to control the threshold

# Define a function to apply a high-pass filter using FFT
def apply_high_pass_filter(image, threshold):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform FFT
    f = np.fft.fft2(gray_image)
    fshift = np.fft.fftshift(f)
    
    # Create a mask with high-pass filter
    rows, cols = gray_image.shape
    crow, ccol = rows // 2, cols // 2  # center
    mask = np.ones((rows, cols), np.uint8)
    mask[crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = 0
    
    # Apply mask and inverse FFT
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    return img_back

# Define a function to calculate the brightness of an image
def calculate_brightness(image):
    # Calculate the average brightness
    brightness = np.mean(image)
    return brightness

# Initialize a list to store brightness values and filenames
brightness_values = []
filenames = []

# Loop through each image in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        filenames.append(filename)
        # Read the image
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Failed to read {filename}")
            continue
        
        # Apply high-pass filter
        high_pass_image = apply_high_pass_filter(image, high_pass_threshold)
        
        # Calculate brightness
        brightness = calculate_brightness(high_pass_image)
        brightness_values.append(brightness)

        # Stop after processing 92 images
        if len(brightness_values) == 92:
            break

# Print the list of filenames
print("Processed filenames:")
print(filenames)

# Data for x-axis (92 values)
x_data = [
    -25.77, -25.53, -25.29, -25.04, -24.8, -24.56, -24.31, -24.07, -23.83, -23.58,
    -23.34, -23.1, -22.86, -22.61, -22.37, -22.13, -21.88, -21.64, -21.4, -21.16,
    -20.91, -20.67, -20.43, -20.19, -19.94, -19.7, -19.46, -19.22, -18.97, -18.73,
    -18.49, -18.25, -18, -17.76, -17.52, -17.28, -17.03, -16.79, -16.55, -16.31,
    -16.07, -15.82, -15.58, -15.34, -15.1, -14.86, -14.61, -14.37, -14.13, -13.88,
    -13.64, -13.4, -13.15, -12.91, -12.67, -12.42, -12.18, -11.94, -11.69, -11.45,
    -11.21, -10.96, -10.72, -10.48, -10.23, -9.99, -9.75, -9.5, -9.26, -9.02,
    -8.77, -8.53, -8.29, -8.04, -7.8, -7.56, -7.31, -7.07, -6.83, -6.58, -6.34,
    -6.1, -5.85, -5.61, -5.37, -5.12, -4.88, -4.64, -4.39, -4.15, -3.91, -3.66, -3.42
]

# Ensure the number of x_data matches the number of brightness values
if len(x_data) != len(brightness_values):
    print("Warning: The number of x-axis values does not match the number of brightness values.")
else:
    # Plot the brightness values
    plt.figure(figsize=(10, 5))
    plt.plot(x_data, brightness_values, marker='o')
    plt.title('Brightness Values of High-Pass Filtered Images')
    plt.xlabel('X Data')
    plt.ylabel('Brightness')
    plt.grid(True)
    plt.show()






sys.exit()



def smooth_data(data, window_size=5):
    """Smooth the data using a simple moving average."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def plot_excel_data(file_path, column_names, x_column, use_dots=True, smooth=False):
    # Read the Excel file
    df = pd.read_excel(file_path)

    # Create a scaler for normalization
    scaler = MinMaxScaler()

    # Create x-axis data
    if x_column and x_column in df.columns:
        x_data = df[x_column].values
    else:
        x_data = range(len(df))

    # Plot for each specified column
    plt.figure(figsize=(10, 5))

    for column_name in column_names:
        # Check if the specified column exists
        if column_name not in df.columns:
            print(f"Column '{column_name}' does not exist in the Excel file.")
            continue

        # Extract the data for the specified column
        data = df[column_name].values

        # Normalize the data
        data_normalized = scaler.fit_transform(data.reshape(-1, 1)).flatten()

        # Smooth the data if requested
        if smooth:
            data_normalized = smooth_data(data_normalized)

            # Adjust x_data for the smoothed data
            x_data = x_data[len(x_data) - len(data_normalized):]

        # Plot the data with or without dots
        if use_dots:
            plt.plot(x_data, data_normalized, linestyle='-', marker='o', alpha=0.7, label=column_name)
        else:
            plt.plot(x_data, data_normalized, linestyle='-', alpha=0.7, label=column_name)

    plt.title('Analisa Perubahan Langit saat Fajar')
    plt.xlabel(x_column if x_column else 'Total Data Points')
    plt.ylabel('Normalized Values')
    plt.grid()
    plt.legend()
    plt.show()

# Example usage
path = os.path.join(os.getcwd(), 'Desktop', 'average_fft_values.xlsx')
file_path = path  # Replace with your Excel file path
column_names = ['Channels Red and Green'] #, 'Average Hue', 'Sum of Red and Green']  # Replace with the columns you want to plot
x_column = 'Elevation'  # Replace with the column you want to use for the x-axis (optional)
plot_excel_data(file_path, column_names, x_column=x_column, use_dots=True, smooth=False) 