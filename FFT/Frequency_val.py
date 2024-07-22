import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def edge_detection(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 100, 200)
    return edges

def calculate_fft(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # Added 1 to avoid log(0)
    return fshift, magnitude_spectrum

def calculate_edge_frequencies(image_path):
    edges = edge_detection(image_path)
    fshift, _ = calculate_fft(edges)

    # Define the region for high frequencies that correspond to edges
    height, width = edges.shape
    # For example, you can focus on the top right quadrant for edges
    high_freq_region = fshift[height//2:, width//2:]  # Adjust this as needed

    return high_freq_region

def calculate_average_edge_frequency(image_path):
    high_freqs = calculate_edge_frequencies(image_path)
    avg_edge_frequency = np.mean(np.abs(high_freqs))  # Average of high frequency magnitudes
    return avg_edge_frequency

def plot_fft_values(fft_values):
    plt.figure()
    plt.title("Average Edge Frequency Values Over Time")
    plt.xlabel("Image Index")
    plt.ylabel("Average Edge Frequency Value")
    plt.plot(fft_values, marker='o')
    plt.show()

def save_to_excel(image_names, fft_values, file_name="average_edge_frequency_values.xlsx"):
    df = pd.DataFrame({'Image Name': image_names, 'Average Edge Frequency Value': fft_values})
    df.to_excel(file_name, index=False)
    print(f"Saved average edge frequency values to {file_name}")

def main(image_directory):
    fft_values = []
    image_names = []

    for image_name in sorted(os.listdir(image_directory)):
        image_path = os.path.join(image_directory, image_name)
        avg_edge_frequency = calculate_average_edge_frequency(image_path)
        fft_values.append(avg_edge_frequency)
        image_names.append(image_name)
        print(f"Average edge frequency value for {image_name}: {avg_edge_frequency}")

    plot_fft_values(fft_values)
    save_to_excel(image_names, fft_values)

if __name__ == "__main__":
    # Directory containing the images
    folder_name = "Desktop/images"
    image_directory = os.path.join(os.getcwd(), folder_name)
    main(image_directory)